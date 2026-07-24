## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00049488


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0014753, 0.0014753)
1: (-0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003676, 0.0003676)
2: (0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0019481, 0.0019481)
3: (-0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0008867, 0.0008867)
4: (0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003770, 0.0003770)
5: (0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0024502, 0.0024502)
6: (-0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0006219, 0.0006219)
7: (-0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0016090, 0.0016090)
8: (-0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0008461, 0.0008461)
9: (0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0009812, 0.0009812)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.86 + 1.43 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0006908, upper bound: 0.0006908

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005762, upper bound: 0.0005762
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005762, upper bound: 0.0005762
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.34 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 0, lower bound: -0.0005762, upper bound: 0.0005762
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 0, lower bound: -0.0005762, upper bound: 0.0005762

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0014712, 0.0014600
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003666, 0.0003638
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0019279, 0.0019428
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0008843, 0.0008775
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003731, 0.0003760
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0024248, 0.0024435
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0006202, 0.0006154
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0016046, 0.0015923
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0008438, 0.0008374
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0009710, 0.0009785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005636, upper bound: 0.0005380
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005380, upper bound: 0.0005636
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0014753, 0.0014712
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003676, 0.0003666
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0019428, 0.0019481
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0008867, 0.0008843
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003760, 0.0003770
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0024435, 0.0024502
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0006219, 0.0006202
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0016090, 0.0016046
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0008461, 0.0008438
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0009785, 0.0009812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005636, upper bound: 0.0005380
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005380, upper bound: 0.0005636
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.05
Output dim: 0, lower bound: -0.0005636, upper bound: 0.0005380
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.05
Output dim: 0, lower bound: -0.0005380, upper bound: 0.0005636
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.05
Output dim: 0, lower bound: -0.0005636, upper bound: 0.0005380
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.05
Output dim: 0, lower bound: -0.0005380, upper bound: 0.0005636

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0014159, 0.0013984
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003528, 0.0003484
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0018466, 0.0018696
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0008510, 0.0008405
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003574, 0.0003619
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0023225, 0.0023515
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005968, 0.0005895
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0015442, 0.0015252
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0008121, 0.0008021
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0009300, 0.0009417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0005160
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005071
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0014097, 0.0014053
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003513, 0.0003502
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0018557, 0.0018614
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0008472, 0.0008446
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003592, 0.0003603
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0023340, 0.0023412
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005942, 0.0005924
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0015374, 0.0015327
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0008085, 0.0008060
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0009346, 0.0009375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005420
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005160, upper bound: 0.0005273
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0014203, 0.0014097
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003539, 0.0003513
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0018614, 0.0018756
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0008537, 0.0008472
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003603, 0.0003630
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0023412, 0.0023590
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005987, 0.0005942
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0015491, 0.0015374
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0008147, 0.0008085
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0009375, 0.0009446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0005160
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005071
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0014141, 0.0014159
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003524, 0.0003528
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0018696, 0.0018673
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0008499, 0.0008510
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003619, 0.0003614
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0023515, 0.0023486
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005961, 0.0005968
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0015423, 0.0015442
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0008111, 0.0008121
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0009417, 0.0009405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005420
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005160, upper bound: 0.0005273
time: 0.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0005160
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005071
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005420
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -0.0005160, upper bound: 0.0005273
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -0.0005273, upper bound: 0.0005160
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -0.0005420, upper bound: 0.0005071
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005420
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -0.0005160, upper bound: 0.0005273

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0013063, 0.0013070
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003255, 0.0003257
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0017259, 0.0017250
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0007852, 0.0007856
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003340, 0.0003339
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0021707, 0.0021696
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005507, 0.0005510
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0014248, 0.0014255
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0007493, 0.0007496
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0008693, 0.0008688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004232, upper bound: 0.0004162
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004232, upper bound: 0.0004162
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0013252, 0.0012889
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003302, 0.0003212
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0017020, 0.0017499
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0007965, 0.0007747
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003294, 0.0003387
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0021406, 0.0022009
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005586, 0.0005433
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0014453, 0.0014057
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0007601, 0.0007393
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0008572, 0.0008813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004378, upper bound: 0.0004112
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004378, upper bound: 0.0004112
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0013001, 0.0013143
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003240, 0.0003275
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0017355, 0.0017168
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0007814, 0.0007899
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003359, 0.0003323
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0021828, 0.0021593
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005481, 0.0005540
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0014180, 0.0014334
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0007457, 0.0007538
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0008741, 0.0008647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004112, upper bound: 0.0004378
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004112, upper bound: 0.0004378
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0013196, 0.0012958
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003288, 0.0003229
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0017111, 0.0017426
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0007931, 0.0007788
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003312, 0.0003373
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0021521, 0.0021917
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005563, 0.0005462
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0014393, 0.0014132
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0007569, 0.0007432
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0008618, 0.0008777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004162, upper bound: 0.0004232
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004162, upper bound: 0.0004232
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0013107, 0.0013196
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003266, 0.0003288
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0017426, 0.0017308
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0007878, 0.0007931
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003373, 0.0003350
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0021917, 0.0021768
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005525, 0.0005563
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0014295, 0.0014393
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0007518, 0.0007569
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0008777, 0.0008717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004232, upper bound: 0.0004162
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004232, upper bound: 0.0004162
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0013295, 0.0013001
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003313, 0.0003240
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0017168, 0.0017556
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0007991, 0.0007814
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003323, 0.0003398
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0021593, 0.0022081
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005604, 0.0005481
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0014500, 0.0014180
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0007626, 0.0007457
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0008647, 0.0008842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004378, upper bound: 0.0004112
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004378, upper bound: 0.0004112
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0013045, 0.0013252
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003250, 0.0003302
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0017499, 0.0017226
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0007840, 0.0007965
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003387, 0.0003334
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0022009, 0.0021665
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005499, 0.0005586
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0014227, 0.0014453
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0007482, 0.0007601
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0008813, 0.0008676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004112, upper bound: 0.0004378
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004112, upper bound: 0.0004378
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9874257, 0.9898434, 0.9874257, 0.9898434, -0.0013240, 0.0013063
1: -0.0043971, -0.0037947, -0.0043971, -0.0037947, -0.0003299, 0.0003255
2: 0.0100561, 0.0132485, 0.0100561, 0.0132485, -0.0017250, 0.0017483
3: -0.0073033, -0.0058502, -0.0073033, -0.0058502, -0.0007958, 0.0007852
4: 0.0024742, 0.0030921, 0.0024742, 0.0030921, -0.0003339, 0.0003384
5: 0.0116073, 0.0156226, 0.0116073, 0.0156226, -0.0021696, 0.0021989
6: -0.0024244, -0.0014052, -0.0024244, -0.0014052, -0.0005581, 0.0005507
7: -0.0094102, -0.0067734, -0.0094102, -0.0067734, -0.0014440, 0.0014248
8: -0.0045129, -0.0031262, -0.0045129, -0.0031262, -0.0007594, 0.0007493
9: 0.0017611, 0.0033690, 0.0017611, 0.0033690, -0.0008688, 0.0008805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004162, upper bound: 0.0004232
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004162, upper bound: 0.0004232
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004232, upper bound: 0.0004162
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004232, upper bound: 0.0004162
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004378, upper bound: 0.0004112
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004378, upper bound: 0.0004112
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004112, upper bound: 0.0004378
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004112, upper bound: 0.0004378
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004162, upper bound: 0.0004232
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004162, upper bound: 0.0004232
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004232, upper bound: 0.0004162
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004232, upper bound: 0.0004162
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004378, upper bound: 0.0004112
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004378, upper bound: 0.0004112
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004112, upper bound: 0.0004378
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004112, upper bound: 0.0004378
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004162, upper bound: 0.0004232
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 0, lower bound: -0.0004162, upper bound: 0.0004232

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.29 + 44.10 = 47.40 seconds
