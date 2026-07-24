## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00221263


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027846, 0.0027846)
1: (0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006474, 0.0006474)
2: (0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016356, 0.0016356)
3: (-0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0016028, 0.0016028)
4: (-0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014518, 0.0014518)
5: (0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015467, 0.0015467)
6: (-0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0055441, 0.0055441)
7: (0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0076037, 0.0076037)
8: (0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0051908, 0.0051908)
9: (-0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0047602, 0.0047602)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.85 + 1.65 = 3.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0025772, upper bound: 0.0025772

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024855, upper bound: 0.0024634
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024633, upper bound: 0.0024855
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 8, lower bound: -0.0024855, upper bound: 0.0024634
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 8, lower bound: -0.0024633, upper bound: 0.0024855

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027721, 0.0027617
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006456, 0.0006441
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016236, 0.0016293
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015896, 0.0015956
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014440, 0.0014376
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015328, 0.0015389
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0054916, 0.0055158
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0075639, 0.0075309
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0051635, 0.0051403
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0047136, 0.0047347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023550, upper bound: 0.0023460
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023557, upper bound: 0.0023448
time: 0.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027617, 0.0027721
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006441, 0.0006456
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016293, 0.0016236
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015956, 0.0015896
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014376, 0.0014440
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015389, 0.0015328
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0055158, 0.0054916
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0075309, 0.0075639
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0051403, 0.0051635
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0047347, 0.0047136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023836, upper bound: 0.0024232
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024066, upper bound: 0.0024054
time: 0.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 8, lower bound: -0.0023550, upper bound: 0.0023460
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 8, lower bound: -0.0023557, upper bound: 0.0023448
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 8, lower bound: -0.0023836, upper bound: 0.0024232
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 8, lower bound: -0.0024066, upper bound: 0.0024054

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027390, 0.0027343
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006405, 0.0006399
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016084, 0.0016110
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015741, 0.0015768
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014237, 0.0014208
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015167, 0.0015195
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0054280, 0.0054390
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0074602, 0.0074452
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0050898, 0.0050793
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0046590, 0.0046685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021274, upper bound: 0.0022063
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022167, upper bound: 0.0021190
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027433, 0.0027286
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006412, 0.0006390
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016053, 0.0016134
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015708, 0.0015792
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014263, 0.0014172
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015134, 0.0015220
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0054148, 0.0054489
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0074737, 0.0074272
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0050993, 0.0050666
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0046475, 0.0046772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022277, upper bound: 0.0022736
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022821, upper bound: 0.0022133
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027477, 0.0027602
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006415, 0.0006433
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016228, 0.0016159
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015869, 0.0015797
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014272, 0.0014349
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015313, 0.0015240
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0054830, 0.0054539
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0074784, 0.0075180
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0051025, 0.0051305
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0047050, 0.0046797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022619, upper bound: 0.0022940
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022626, upper bound: 0.0022937
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027490, 0.0027581
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006417, 0.0006430
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016216, 0.0016166
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015857, 0.0015804
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014280, 0.0014336
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015301, 0.0015248
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0054781, 0.0054569
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0074825, 0.0075113
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0051054, 0.0051257
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0047007, 0.0046823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024040, upper bound: 0.0023899
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023949, upper bound: 0.0024027
time: 0.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0021274, upper bound: 0.0022063
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0022167, upper bound: 0.0021190
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0022277, upper bound: 0.0022736
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0022821, upper bound: 0.0022133
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0022619, upper bound: 0.0022940
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0022626, upper bound: 0.0022937
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0024040, upper bound: 0.0023899
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0023949, upper bound: 0.0024027

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0023463, 0.0022909
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0005899, 0.0005819
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0013664, 0.0013970
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0013348, 0.0013664
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0011873, 0.0011531
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0012525, 0.0012849
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0043831, 0.0045117
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0062593, 0.0060842
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0042199, 0.0040965
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0037843, 0.0038963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019399, upper bound: 0.0018519
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019399, upper bound: 0.0018519
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026105, 0.0026074
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006154, 0.0006150
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015449, 0.0015466
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0014781, 0.0014798
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013166, 0.0013147
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014325, 0.0014343
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0050414, 0.0050485
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0069125, 0.0069027
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0047072, 0.0047003
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0043120, 0.0043183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019570, upper bound: 0.0020051
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019570, upper bound: 0.0020051
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026195, 0.0025958
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006167, 0.0006133
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015385, 0.0015515
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0014714, 0.0014849
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013221, 0.0013075
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014257, 0.0014396
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0050144, 0.0050693
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0069407, 0.0068660
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0047271, 0.0046744
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0042886, 0.0043364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021562
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022206, upper bound: 0.0021289
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027150, 0.0027320
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006364, 0.0006389
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016072, 0.0015978
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015709, 0.0015612
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014069, 0.0014174
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015148, 0.0015048
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0054173, 0.0053778
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0073742, 0.0074280
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0050295, 0.0050674
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0046480, 0.0046136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022410, upper bound: 0.0022725
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022412, upper bound: 0.0022725
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027207, 0.0027275
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006372, 0.0006382
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016047, 0.0016009
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015684, 0.0015645
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0014104, 0.0014146
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015122, 0.0015082
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0054069, 0.0053910
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0073922, 0.0074139
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0050422, 0.0050575
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0046390, 0.0046252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022602, upper bound: 0.0022847
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022481, upper bound: 0.0022912
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0027166, 0.0027063
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006316, 0.0006301
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015957, 0.0016013
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015356, 0.0015415
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013938, 0.0013875
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014949, 0.0015009
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0052988, 0.0053227
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0072993, 0.0072668
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0049803, 0.0049573
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0045496, 0.0045704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021763, upper bound: 0.0022510
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022674, upper bound: 0.0021522
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026972, 0.0027251
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006288, 0.0006328
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0016060, 0.0015906
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015463, 0.0015304
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013818, 0.0013991
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015059, 0.0014896
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0053424, 0.0052776
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0072380, 0.0073262
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0049370, 0.0049992
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0045876, 0.0045312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022218, upper bound: 0.0022917
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022942, upper bound: 0.0022314
time: 0.66 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0019399, upper bound: 0.0018519
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0019399, upper bound: 0.0018519
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0019570, upper bound: 0.0020051
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0019570, upper bound: 0.0020051
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0021975, upper bound: 0.0021562
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0022206, upper bound: 0.0021289
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0022410, upper bound: 0.0022725
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0022412, upper bound: 0.0022725
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0022602, upper bound: 0.0022847
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0022481, upper bound: 0.0022912
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0021763, upper bound: 0.0022510
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0022674, upper bound: 0.0021522
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0022218, upper bound: 0.0022917
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 8, lower bound: -0.0022942, upper bound: 0.0022314

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026075, 0.0025816
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006144, 0.0006107
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015309, 0.0015452
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0014609, 0.0014757
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013133, 0.0012972
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014176, 0.0014327
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0049781, 0.0050384
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0068961, 0.0068140
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0046954, 0.0046376
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0042551, 0.0043076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020442, upper bound: 0.0020094
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021137, upper bound: 0.0019751
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026896, 0.0027047
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006332, 0.0006354
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015915, 0.0015831
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015573, 0.0015486
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013933, 0.0014027
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014993, 0.0014905
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0053596, 0.0053243
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0073045, 0.0073525
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0049801, 0.0050140
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0045998, 0.0045691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021099, upper bound: 0.0022001
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021677, upper bound: 0.0021396
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026877, 0.0027069
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006330, 0.0006357
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015927, 0.0015821
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015585, 0.0015476
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013921, 0.0014040
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0015006, 0.0014894
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0053646, 0.0053201
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0072988, 0.0073594
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0049761, 0.0050188
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0046041, 0.0045654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020141, upper bound: 0.0021331
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021003, upper bound: 0.0020419
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026879, 0.0026770
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006269, 0.0006253
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015794, 0.0015855
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015178, 0.0015240
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013760, 0.0013692
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014777, 0.0014841
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0052306, 0.0052561
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0072076, 0.0071730
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0049164, 0.0048920
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0044898, 0.0045120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021293, upper bound: 0.0022093
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021862, upper bound: 0.0021557
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026701, 0.0026948
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006243, 0.0006279
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015893, 0.0015757
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015280, 0.0015138
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013650, 0.0013803
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014882, 0.0014737
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0052722, 0.0052147
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0071514, 0.0072295
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0048767, 0.0049318
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0045260, 0.0044760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022273, upper bound: 0.0022700
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022273, upper bound: 0.0022700
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0022158, 0.0022561
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0005664, 0.0005723
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0013559, 0.0013336
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0012917, 0.0012686
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0010801, 0.0011050
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0012252, 0.0012016
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0042331, 0.0041394
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0057053, 0.0058330
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0038461, 0.0039360
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0036268, 0.0035452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021166, upper bound: 0.0021850
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021166, upper bound: 0.0021851
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0022533, 0.0022055
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0005719, 0.0005650
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0013279, 0.0013543
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0012628, 0.0012901
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0011033, 0.0010737
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0011956, 0.0012235
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0041155, 0.0042265
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0058240, 0.0056728
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0039297, 0.0038232
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0035244, 0.0036210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022062, upper bound: 0.0020906
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022062, upper bound: 0.0020906
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0025413, 0.0025801
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006055, 0.0006111
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015259, 0.0015044
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0014615, 0.0014393
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0012852, 0.0013092
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014210, 0.0013982
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0050055, 0.0049153
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0067438, 0.0068666
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0045895, 0.0046760
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0042938, 0.0042153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021007, upper bound: 0.0021600
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021009, upper bound: 0.0021562
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0025538, 0.0025692
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006073, 0.0006095
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015199, 0.0015113
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0014553, 0.0014464
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0012929, 0.0013025
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014146, 0.0014055
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0049802, 0.0049442
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0067832, 0.0068321
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0046172, 0.0046517
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0042717, 0.0042405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021673, upper bound: 0.0020984
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021706, upper bound: 0.0020975
time: 0.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0020442, upper bound: 0.0020094
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021137, upper bound: 0.0019751
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021099, upper bound: 0.0022001
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021677, upper bound: 0.0021396
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0020141, upper bound: 0.0021331
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021003, upper bound: 0.0020419
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021293, upper bound: 0.0022093
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021862, upper bound: 0.0021557
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0022273, upper bound: 0.0022700
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0022273, upper bound: 0.0022700
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021166, upper bound: 0.0021850
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021166, upper bound: 0.0021851
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0022062, upper bound: 0.0020906
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0022062, upper bound: 0.0020906
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021007, upper bound: 0.0021600
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021009, upper bound: 0.0021562
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021673, upper bound: 0.0020984
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.03
Output dim: 8, lower bound: -0.0021706, upper bound: 0.0020975

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026452, 0.0026690
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006210, 0.0006244
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015750, 0.0015619
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015140, 0.0015004
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013504, 0.0013652
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014733, 0.0014594
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0052160, 0.0051607
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0070771, 0.0071524
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0048244, 0.0048775
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0044767, 0.0044285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021033, upper bound: 0.0021977
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021516, upper bound: 0.0021362
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0059729, 0.0092381, 0.0059729, 0.0092381, -0.0026443, 0.0026712
1: 0.0019033, 0.0026170, 0.0019033, 0.0026170, -0.0006208, 0.0006247
2: 0.0091959, 0.0110576, 0.0091959, 0.0110576, -0.0015763, 0.0015614
3: -0.0049533, -0.0029858, -0.0049533, -0.0029858, -0.0015153, 0.0014999
4: -0.0005501, 0.0013252, -0.0005501, 0.0013252, -0.0013499, 0.0013666
5: 0.0027488, 0.0046102, 0.0027488, 0.0046102, -0.0014746, 0.0014588
6: -0.0110670, -0.0040086, -0.0110670, -0.0040086, -0.0052212, 0.0051586
7: 0.0026133, 0.0123641, 0.0026133, 0.0123641, -0.0070742, 0.0071595
8: 0.9912586, 0.9979576, 0.9912586, 0.9979576, -0.0048224, 0.0048825
9: -0.0140023, -0.0078583, -0.0140023, -0.0078583, -0.0044812, 0.0044267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020598, upper bound: 0.0021644
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021179, upper bound: 0.0020976
time: 0.69 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.97 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.0021033, upper bound: 0.0021977
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.0021516, upper bound: 0.0021362
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.0020598, upper bound: 0.0021644
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.0021179, upper bound: 0.0020976

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.49 + 85.88 = 89.38 seconds
