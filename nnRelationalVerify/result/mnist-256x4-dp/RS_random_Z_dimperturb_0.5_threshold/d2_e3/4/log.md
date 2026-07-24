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
0: (-0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802)
1: (-0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0064725, 0.0064725)
2: (0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0050531, 0.0050531)
3: (-0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425)
4: (0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0056115, 0.0056115)
5: (-0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0054727, 0.0054727)
6: (-0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033811, 0.0033811)
7: (-0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056442, 0.0056442)
8: (-0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197)
9: (0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0112994, 0.0112994)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.66 = 3.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0064889, upper bound: 0.0064889

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0064186, upper bound: 0.0064186
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0064186, upper bound: 0.0064186
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 9, lower bound: -0.0064186, upper bound: 0.0064186
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 9, lower bound: -0.0064186, upper bound: 0.0064186

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0064678, 0.0064683
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0050423, 0.0050430
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0056080, 0.0056075
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0054591, 0.0054614
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033809, 0.0033809
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056188, 0.0056165
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0112935, 0.0112929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059768, upper bound: 0.0060848
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060848, upper bound: 0.0059768
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0064683, 0.0064725
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0050430, 0.0050531
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0056075, 0.0056115
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0054727, 0.0054591
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033811, 0.0033809
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056442, 0.0056188
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0112929, 0.0112994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0062906, upper bound: 0.0062318
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0062318, upper bound: 0.0062906
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 9, lower bound: -0.0059768, upper bound: 0.0060848
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 9, lower bound: -0.0060848, upper bound: 0.0059768
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 9, lower bound: -0.0062906, upper bound: 0.0062318
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 9, lower bound: -0.0062318, upper bound: 0.0062906

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032535, 0.0032979
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0051121, 0.0050084
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0044298, 0.0043592
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055180, 0.0055518
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053346, 0.0052798
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033788, 0.0033695
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055196, 0.0055456
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008826, 0.0008744
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0103302, 0.0104732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058499, upper bound: 0.0059415
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058062, upper bound: 0.0059643
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032982, 0.0032529
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0050079, 0.0051122
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043585, 0.0044300
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055519, 0.0055175
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052775, 0.0053354
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033695, 0.0033788
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055477, 0.0055173
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008743, 0.0008829
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0104727, 0.0103295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060709, upper bound: 0.0058342
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059641, upper bound: 0.0059629
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0064354, 0.0064475
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0050117, 0.0050292
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0056016, 0.0056041
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0054688, 0.0054562
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033778, 0.0033783
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056229, 0.0055900
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009162
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0112593, 0.0112579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0061391, upper bound: 0.0060648
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0061051, upper bound: 0.0060903
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0064433, 0.0064397
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0050192, 0.0050218
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0056001, 0.0056056
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0054697, 0.0054553
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033784, 0.0033777
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056153, 0.0055975
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009155
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0112513, 0.0112658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060903, upper bound: 0.0061051
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060648, upper bound: 0.0061392
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 9, lower bound: -0.0058499, upper bound: 0.0059415
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 9, lower bound: -0.0058062, upper bound: 0.0059643
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 9, lower bound: -0.0060709, upper bound: 0.0058342
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 9, lower bound: -0.0059641, upper bound: 0.0059629
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 9, lower bound: -0.0061391, upper bound: 0.0060648
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 9, lower bound: -0.0061051, upper bound: 0.0060903
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 9, lower bound: -0.0060903, upper bound: 0.0061051
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 9, lower bound: -0.0060648, upper bound: 0.0061392

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031911, 0.0032295
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0050014, 0.0049079
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043794, 0.0043157
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054886, 0.0055176
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052706, 0.0052217
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033713, 0.0033631
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055103, 0.0055290
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008567, 0.0008391
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102107, 0.0103346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058352, upper bound: 0.0058236
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056999, upper bound: 0.0059272
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031765, 0.0032355
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0050116, 0.0048821
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043863, 0.0043014
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054773, 0.0055224
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052764, 0.0052065
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033724, 0.0033605
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055030, 0.0055341
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008473, 0.0008488
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101683, 0.0103537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056169, upper bound: 0.0052548
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050565, upper bound: 0.0057688
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0033247, 0.0032706
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0049386, 0.0050746
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042075, 0.0043028
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055235, 0.0054819
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052378, 0.0053022
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033583, 0.0033701
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054908, 0.0054505
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008733, 0.0008822
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102954, 0.0101175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058956, upper bound: 0.0052369
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053820, upper bound: 0.0056294
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0033159, 0.0032803
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0049683, 0.0050429
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042304, 0.0042791
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055163, 0.0054897
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052465, 0.0052956
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033608, 0.0033676
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054810, 0.0054616
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008735, 0.0008818
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102607, 0.0101541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058424, upper bound: 0.0057910
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058236, upper bound: 0.0058353
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0063096, 0.0063474
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049509, 0.0049827
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055710, 0.0055621
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053935, 0.0053965
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033689, 0.0033720
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056133, 0.0055731
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009035, 0.0008808
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0111325, 0.0110882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056913, upper bound: 0.0057528
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058082, upper bound: 0.0055987
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0063354, 0.0063217
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049651, 0.0049679
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055599, 0.0055736
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0054092, 0.0053810
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033715, 0.0033694
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056061, 0.0055804
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008936, 0.0008901
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0110905, 0.0111311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060909, upper bound: 0.0059465
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059760, upper bound: 0.0060758
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0063176, 0.0063396
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049577, 0.0049753
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055696, 0.0055639
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053945, 0.0053956
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033696, 0.0033714
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056058, 0.0055807
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009036, 0.0008802
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0111246, 0.0110971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060758, upper bound: 0.0059760
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059464, upper bound: 0.0060909
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0063433, 0.0063138
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049725, 0.0049611
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055581, 0.0055750
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0054101, 0.0053800
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033721, 0.0033688
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055985, 0.0055879
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008943, 0.0008900
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0110817, 0.0111390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060526, upper bound: 0.0060422
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059705, upper bound: 0.0061269
time: 0.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0058352, upper bound: 0.0058236
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0056999, upper bound: 0.0059272
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0056169, upper bound: 0.0052548
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0050565, upper bound: 0.0057688
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0058956, upper bound: 0.0052369
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0053820, upper bound: 0.0056294
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0058424, upper bound: 0.0057910
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0058236, upper bound: 0.0058353
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0056913, upper bound: 0.0057528
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0058082, upper bound: 0.0055987
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0060909, upper bound: 0.0059465
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0059760, upper bound: 0.0060758
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0060758, upper bound: 0.0059760
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0059464, upper bound: 0.0060909
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0060526, upper bound: 0.0060422
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 9, lower bound: -0.0059705, upper bound: 0.0061269

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032193, 0.0032477
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0049256, 0.0048620
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042304, 0.0041899
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054613, 0.0054823
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052312, 0.0051913
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033601, 0.0033544
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054551, 0.0054618
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008563, 0.0008385
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100383, 0.0101253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056296, upper bound: 0.0051579
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051393, upper bound: 0.0056171
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032094, 0.0032558
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0049525, 0.0048321
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042521, 0.0041666
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054533, 0.0054892
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052382, 0.0051822
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033623, 0.0033519
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054431, 0.0054713
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008562, 0.0008381
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100014, 0.0101578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056876, upper bound: 0.0058245
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056191, upper bound: 0.0059141
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031361, 0.0031174
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042476, 0.0043004
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037448, 0.0037774
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052735, 0.0052591
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048524, 0.0048782
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033043, 0.0033088
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053984, 0.0054037
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007585, 0.0007751
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091344, 0.0090681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053647, upper bound: 0.0049918
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053465, upper bound: 0.0050637
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030584, 0.0031938
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0044337, 0.0041181
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038662, 0.0036599
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052140, 0.0053177
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049396, 0.0047825
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033208, 0.0032924
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053726, 0.0054285
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007729, 0.0007600
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088827, 0.0093186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048902, upper bound: 0.0055021
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048083, upper bound: 0.0055158
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032806, 0.0031497
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042327, 0.0045540
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035566, 0.0037708
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053175, 0.0052175
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048119, 0.0049673
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032899, 0.0033182
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053863, 0.0053218
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007831, 0.0008060
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092637, 0.0088345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057545, upper bound: 0.0050065
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057454, upper bound: 0.0050883
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032038, 0.0032225
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0044055, 0.0043687
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036686, 0.0036519
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052590, 0.0052729
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049018, 0.0048763
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033053, 0.0033017
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053621, 0.0053478
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007969, 0.0007920
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090124, 0.0090682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052447, upper bound: 0.0054370
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051426, upper bound: 0.0054879
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032540, 0.0032037
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048359, 0.0049359
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0041751, 0.0042374
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054873, 0.0054494
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051734, 0.0052378
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033517, 0.0033611
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054688, 0.0054445
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008477, 0.0008466
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101439, 0.0099947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056834, upper bound: 0.0055847
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056325, upper bound: 0.0056386
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032484, 0.0032184
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048614, 0.0049256
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0041888, 0.0042304
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054826, 0.0054606
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051887, 0.0052339
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033544, 0.0033601
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054639, 0.0054521
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008384, 0.0008567
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101250, 0.0100374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058106, upper bound: 0.0057473
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057113, upper bound: 0.0058222
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031865, 0.0032324
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0049695, 0.0048878
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043470, 0.0043010
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054820, 0.0055139
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052799, 0.0052162
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033682, 0.0033605
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055146, 0.0055035
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008635, 0.0008315
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101759, 0.0102994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056769, upper bound: 0.0056214
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055358, upper bound: 0.0057385
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032324, 0.0031784
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048502, 0.0049941
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042690, 0.0043715
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055170, 0.0054730
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052129, 0.0052736
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033574, 0.0033700
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055378, 0.0054754
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008550, 0.0008398
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0103231, 0.0101320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055158, upper bound: 0.0048083
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050637, upper bound: 0.0053465
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0062925, 0.0063092
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0048420, 0.0048689
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055370, 0.0055427
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053705, 0.0053514
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033617, 0.0033620
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055605, 0.0055217
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008931, 0.0008897
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0109413, 0.0109455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056386, upper bound: 0.0056325
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057786, upper bound: 0.0054902
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0063223, 0.0062789
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0048653, 0.0048455
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055291, 0.0055505
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053793, 0.0053422
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033642, 0.0033595
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055481, 0.0055337
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008933, 0.0008897
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0109050, 0.0109821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057062, upper bound: 0.0053030
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053033, upper bound: 0.0058011
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0062746, 0.0063266
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0048346, 0.0048763
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055465, 0.0055331
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053559, 0.0053656
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033597, 0.0033640
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055598, 0.0055220
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009033, 0.0008797
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0109756, 0.0109115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060634, upper bound: 0.0058781
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059802, upper bound: 0.0059636
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0063049, 0.0062968
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0048580, 0.0048529
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055388, 0.0055409
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053650, 0.0053569
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033622, 0.0033615
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055478, 0.0055344
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009032, 0.0008795
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0109390, 0.0109478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054902, upper bound: 0.0057786
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056325, upper bound: 0.0056386
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0062202, 0.0062109
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0050809, 0.0050838
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055599, 0.0055766
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053350, 0.0053144
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033827, 0.0033807
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0058418, 0.0058533
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0111853, 0.0112283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0060387, upper bound: 0.0058986
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059134, upper bound: 0.0060275
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0062403, 0.0061903
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0050946, 0.0050692
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055597, 0.0055768
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053440, 0.0053044
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033841, 0.0033794
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0058662, 0.0058301
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0111709, 0.0112426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059567, upper bound: 0.0059836
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0058390, upper bound: 0.0061123
time: 0.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0056296, upper bound: 0.0051579
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0051393, upper bound: 0.0056171
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0056876, upper bound: 0.0058245
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0056191, upper bound: 0.0059141
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0053647, upper bound: 0.0049918
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0053465, upper bound: 0.0050637
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0048902, upper bound: 0.0055021
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0048083, upper bound: 0.0055158
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0057545, upper bound: 0.0050065
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0057454, upper bound: 0.0050883
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0052447, upper bound: 0.0054370
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0051426, upper bound: 0.0054879
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0056834, upper bound: 0.0055847
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0056325, upper bound: 0.0056386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0058106, upper bound: 0.0057473
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0057113, upper bound: 0.0058222
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0056769, upper bound: 0.0056214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0055358, upper bound: 0.0057385
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0055158, upper bound: 0.0048083
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0050637, upper bound: 0.0053465
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0056386, upper bound: 0.0056325
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0057786, upper bound: 0.0054902
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0057062, upper bound: 0.0053030
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0053033, upper bound: 0.0058011
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0060634, upper bound: 0.0058781
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0059802, upper bound: 0.0059636
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0054902, upper bound: 0.0057786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0056325, upper bound: 0.0056386
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0060387, upper bound: 0.0058986
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0059134, upper bound: 0.0060275
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0059567, upper bound: 0.0059836
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 9, lower bound: -0.0058390, upper bound: 0.0061123

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031822, 0.0031331
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042328, 0.0043504
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035890, 0.0036663
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052578, 0.0052201
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048087, 0.0048610
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032923, 0.0033029
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053513, 0.0053332
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007667, 0.0007626
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090164, 0.0088546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056181, upper bound: 0.0050705
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055376, upper bound: 0.0051456
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031047, 0.0032115
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0044159, 0.0041691
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037102, 0.0035485
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051990, 0.0052803
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049018, 0.0047688
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033087, 0.0032866
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053265, 0.0053589
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007822, 0.0007490
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087676, 0.0091068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049343, upper bound: 0.0053737
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048633, upper bound: 0.0053950
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032181, 0.0032636
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0047864, 0.0046913
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043821, 0.0043132
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054550, 0.0054906
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051632, 0.0051158
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033722, 0.0033637
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056985, 0.0057467
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008927, 0.0009007
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101185, 0.0102537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054690, upper bound: 0.0050922
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050756, upper bound: 0.0056335
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032172, 0.0032650
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048116, 0.0046704
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043988, 0.0042986
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054548, 0.0054909
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051774, 0.0051072
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033741, 0.0033623
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057227, 0.0057267
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0008747
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101031, 0.0102749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054555, upper bound: 0.0057266
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054105, upper bound: 0.0057663
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031319, 0.0031113
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042136, 0.0042741
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037091, 0.0037472
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052674, 0.0052513
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048483, 0.0048751
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033010, 0.0033062
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053761, 0.0053756
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007501, 0.0007674
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090984, 0.0090236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053512, upper bound: 0.0049432
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052420, upper bound: 0.0049767
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031300, 0.0031140
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042243, 0.0042664
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037166, 0.0037418
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052657, 0.0052536
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048496, 0.0048742
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033019, 0.0033056
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053702, 0.0053804
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007509, 0.0007671
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090899, 0.0090351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053336, upper bound: 0.0049854
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052655, upper bound: 0.0050495
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030543, 0.0031877
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0043997, 0.0040921
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038306, 0.0036311
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052080, 0.0053099
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049356, 0.0047797
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033176, 0.0032898
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053520, 0.0054003
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007646, 0.0007523
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088471, 0.0092741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048754, upper bound: 0.0053737
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048421, upper bound: 0.0054882
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030522, 0.0031896
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0044068, 0.0040841
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038352, 0.0036243
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052062, 0.0053114
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049364, 0.0047784
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033181, 0.0032892
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053445, 0.0054055
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007652, 0.0007517
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088382, 0.0092811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047962, upper bound: 0.0054098
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047172, upper bound: 0.0055023
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032250, 0.0030791
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041130, 0.0044601
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035098, 0.0037387
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052906, 0.0051792
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047418, 0.0049128
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032815, 0.0033124
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053733, 0.0053049
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007582, 0.0007715
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091593, 0.0086872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057433, upper bound: 0.0049375
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056457, upper bound: 0.0049939
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032224, 0.0030941
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041388, 0.0044446
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035245, 0.0037311
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052883, 0.0051906
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047574, 0.0049149
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032841, 0.0033110
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053694, 0.0053122
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007486, 0.0007831
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091390, 0.0087300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055018, upper bound: 0.0048307
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054882, upper bound: 0.0048940
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032003, 0.0032163
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0043708, 0.0043436
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036321, 0.0036229
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052534, 0.0052652
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048978, 0.0048736
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033021, 0.0032992
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053384, 0.0053193
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007885, 0.0007843
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089772, 0.0090228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052320, upper bound: 0.0053519
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051873, upper bound: 0.0054245
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031976, 0.0032179
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0043777, 0.0043341
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036376, 0.0036153
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052513, 0.0052665
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048986, 0.0048723
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033026, 0.0032985
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053336, 0.0053263
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007891, 0.0007836
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089669, 0.0090311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049767, upper bound: 0.0052420
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049422, upper bound: 0.0052430
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032509, 0.0031978
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048064, 0.0049167
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0041428, 0.0042123
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054819, 0.0054417
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051693, 0.0052351
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033485, 0.0033588
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054459, 0.0054166
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008401, 0.0008391
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101133, 0.0099526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053950, upper bound: 0.0047935
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050123, upper bound: 0.0053325
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032482, 0.0031998
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048144, 0.0049064
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0041489, 0.0042052
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054796, 0.0054434
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051704, 0.0052337
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033492, 0.0033579
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054409, 0.0054239
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008401, 0.0008384
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101018, 0.0099610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056205, upper bound: 0.0055534
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055220, upper bound: 0.0056262
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032576, 0.0032262
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0047015, 0.0047848
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043219, 0.0043770
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054844, 0.0054622
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051137, 0.0051725
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033648, 0.0033719
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057193, 0.0057324
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008750, 0.0009189
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102421, 0.0101393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056612, upper bound: 0.0055336
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056095, upper bound: 0.0055875
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032562, 0.0032271
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0047205, 0.0047597
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043354, 0.0043594
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054841, 0.0054624
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051223, 0.0051589
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033661, 0.0033701
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057385, 0.0057076
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009023, 0.0008933
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102228, 0.0101545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054959, upper bound: 0.0051263
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050715, upper bound: 0.0056181
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032148, 0.0032503
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048961, 0.0048447
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0041981, 0.0041757
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054547, 0.0054786
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052407, 0.0051856
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033570, 0.0033518
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054601, 0.0054360
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008631, 0.0008308
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100035, 0.0100900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056645, upper bound: 0.0055164
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055875, upper bound: 0.0056095
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032047, 0.0032584
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0049231, 0.0048145
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042204, 0.0041528
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054467, 0.0054854
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052477, 0.0051768
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033592, 0.0033492
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054484, 0.0054458
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008630, 0.0008306
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0099666, 0.0101225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052430, upper bound: 0.0049422
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048940, upper bound: 0.0054882
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031896, 0.0030602
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040841, 0.0044098
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036243, 0.0038450
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053114, 0.0052096
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047888, 0.0049364
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032893, 0.0033181
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054298, 0.0053445
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007660, 0.0007652
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092811, 0.0088443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055018, upper bound: 0.0047649
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053950, upper bound: 0.0047935
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031140, 0.0031382
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042664, 0.0042274
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037418, 0.0037269
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052536, 0.0052693
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048855, 0.0048496
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033057, 0.0033019
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054074, 0.0053702
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007818, 0.0007509
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090351, 0.0090963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050495, upper bound: 0.0052655
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049854, upper bound: 0.0053336
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031998, 0.0032563
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0049064, 0.0048191
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042052, 0.0041614
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054434, 0.0054834
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052465, 0.0051704
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033581, 0.0033492
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054535, 0.0054409
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008528, 0.0008401
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0099610, 0.0101091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053512, upper bound: 0.0049432
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048754, upper bound: 0.0053737
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032520, 0.0032113
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048026, 0.0049373
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0041343, 0.0042386
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054832, 0.0054492
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051894, 0.0052351
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033488, 0.0033598
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054764, 0.0054151
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008444, 0.0008486
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101249, 0.0099655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057663, upper bound: 0.0054105
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056776, upper bound: 0.0054781
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0057867, 0.0059252
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043153, 0.0044118
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053163, 0.0052783
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049498, 0.0050091
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032940, 0.0033058
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054370, 0.0053979
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008035, 0.0008149
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0099714, 0.0097972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056938, upper bound: 0.0052136
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056074, upper bound: 0.0052892
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0059680, 0.0057432
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0044331, 0.0042957
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052569, 0.0053371
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0050431, 0.0049127
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033103, 0.0032894
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054129, 0.0054226
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008175, 0.0007996
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0097197, 0.0100458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052893, upper bound: 0.0057057
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052206, upper bound: 0.0057882
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0061565, 0.0062291
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049505, 0.0050062
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055483, 0.0055347
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052791, 0.0052976
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033704, 0.0033760
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0058165, 0.0058014
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0110781, 0.0110007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057882, upper bound: 0.0052206
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052892, upper bound: 0.0056074
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0061769, 0.0062100
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049639, 0.0049928
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055481, 0.0055349
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052891, 0.0052889
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033717, 0.0033747
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0058436, 0.0057782
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009122
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0110630, 0.0110141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057057, upper bound: 0.0052893
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052136, upper bound: 0.0056938
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032029, 0.0032604
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0049333, 0.0048066
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042275, 0.0041454
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054452, 0.0054871
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052486, 0.0051759
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033600, 0.0033487
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054408, 0.0054507
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008630, 0.0008300
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0099586, 0.0101317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052235, upper bound: 0.0050179
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048307, upper bound: 0.0055018
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032479, 0.0032082
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0048150, 0.0049104
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0041503, 0.0042162
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054795, 0.0054474
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051840, 0.0052329
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033493, 0.0033579
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0054666, 0.0054278
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008545, 0.0008384
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101023, 0.0099678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056205, upper bound: 0.0055536
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055220, upper bound: 0.0056262
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0061825, 0.0062035
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049651, 0.0049927
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055371, 0.0055458
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0052946, 0.0052835
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033729, 0.0033734
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0058097, 0.0058086
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0110354, 0.0110417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055725, upper bound: 0.0055728
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057266, upper bound: 0.0054555
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0062146, 0.0061734
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049899, 0.0049686
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055291, 0.0055539
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053034, 0.0052745
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033755, 0.0033709
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057973, 0.0058232
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0109987, 0.0110789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056738, upper bound: 0.0052984
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052175, upper bound: 0.0057176
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0062026, 0.0061832
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0049787, 0.0049804
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055369, 0.0055460
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053037, 0.0052736
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033743, 0.0033720
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0058366, 0.0057854
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0110198, 0.0110561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057056, upper bound: 0.0053247
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051602, upper bound: 0.0056938
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0062328, 0.0061528
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0050016, 0.0049541
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0055289, 0.0055541
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0053123, 0.0052645
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033768, 0.0033697
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0058217, 0.0057972
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0009197
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0109842, 0.0110930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055948, upper bound: 0.0053624
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051401, upper bound: 0.0058038
time: 0.81 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056181, upper bound: 0.0050705
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0055376, upper bound: 0.0051456
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0049343, upper bound: 0.0053737
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0048633, upper bound: 0.0053950
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0054690, upper bound: 0.0050922
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0050756, upper bound: 0.0056335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0054555, upper bound: 0.0057266
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0054105, upper bound: 0.0057663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0053512, upper bound: 0.0049432
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052420, upper bound: 0.0049767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0053336, upper bound: 0.0049854
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052655, upper bound: 0.0050495
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0048754, upper bound: 0.0053737
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0048421, upper bound: 0.0054882
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0047962, upper bound: 0.0054098
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0047172, upper bound: 0.0055023
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0057433, upper bound: 0.0049375
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056457, upper bound: 0.0049939
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0055018, upper bound: 0.0048307
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0054882, upper bound: 0.0048940
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052320, upper bound: 0.0053519
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0051873, upper bound: 0.0054245
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0049767, upper bound: 0.0052420
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0049422, upper bound: 0.0052430
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0053950, upper bound: 0.0047935
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0050123, upper bound: 0.0053325
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056205, upper bound: 0.0055534
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0055220, upper bound: 0.0056262
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056612, upper bound: 0.0055336
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056095, upper bound: 0.0055875
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0054959, upper bound: 0.0051263
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0050715, upper bound: 0.0056181
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056645, upper bound: 0.0055164
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0055875, upper bound: 0.0056095
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052430, upper bound: 0.0049422
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0048940, upper bound: 0.0054882
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0055018, upper bound: 0.0047649
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0053950, upper bound: 0.0047935
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0050495, upper bound: 0.0052655
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0049854, upper bound: 0.0053336
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0053512, upper bound: 0.0049432
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0048754, upper bound: 0.0053737
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0057663, upper bound: 0.0054105
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056776, upper bound: 0.0054781
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056938, upper bound: 0.0052136
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056074, upper bound: 0.0052892
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052893, upper bound: 0.0057057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052206, upper bound: 0.0057882
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0057882, upper bound: 0.0052206
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052892, upper bound: 0.0056074
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0057057, upper bound: 0.0052893
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052136, upper bound: 0.0056938
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052235, upper bound: 0.0050179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0048307, upper bound: 0.0055018
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056205, upper bound: 0.0055536
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0055220, upper bound: 0.0056262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0055725, upper bound: 0.0055728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0057266, upper bound: 0.0054555
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0056738, upper bound: 0.0052984
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0052175, upper bound: 0.0057176
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0057056, upper bound: 0.0053247
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0051602, upper bound: 0.0056938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0055948, upper bound: 0.0053624
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 9, lower bound: -0.0051401, upper bound: 0.0058038

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031930, 0.0031427
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039638, 0.0041065
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037256, 0.0038207
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052595, 0.0052215
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047359, 0.0047998
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033021, 0.0033144
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055971, 0.0055982
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008237, 0.0008479
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091304, 0.0089490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053382, upper bound: 0.0048286
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053248, upper bound: 0.0048999
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031918, 0.0031441
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039889, 0.0040803
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037434, 0.0038024
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052593, 0.0052218
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047498, 0.0047883
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033039, 0.0033126
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056195, 0.0055790
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008507, 0.0008196
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091098, 0.0089686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052674, upper bound: 0.0049103
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052541, upper bound: 0.0049807
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031006, 0.0032053
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0043813, 0.0041428
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036737, 0.0035190
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051930, 0.0052725
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048978, 0.0047658
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033055, 0.0032840
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053052, 0.0053303
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007746, 0.0007413
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087306, 0.0090613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049196, upper bound: 0.0052656
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048569, upper bound: 0.0053613
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030985, 0.0032068
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0043878, 0.0041345
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036790, 0.0035120
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051913, 0.0052738
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048984, 0.0047648
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033061, 0.0032834
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0052980, 0.0053354
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007746, 0.0007409
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087221, 0.0090692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048497, upper bound: 0.0052845
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047786, upper bound: 0.0053826
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031782, 0.0031508
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039905, 0.0040682
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037485, 0.0037920
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052481, 0.0052283
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047429, 0.0047874
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033042, 0.0033110
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055842, 0.0056085
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008236, 0.0008473
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090788, 0.0089799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052303, upper bound: 0.0048492
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052111, upper bound: 0.0049233
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031053, 0.0032312
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041773, 0.0038954
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038684, 0.0036796
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051928, 0.0052892
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048390, 0.0046956
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033209, 0.0032957
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055603, 0.0056315
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008395, 0.0008316
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088447, 0.0092328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048802, upper bound: 0.0053787
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048172, upper bound: 0.0053957
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032131, 0.0032591
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0047804, 0.0046478
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043675, 0.0042758
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054487, 0.0054832
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051734, 0.0051041
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033709, 0.0033597
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057017, 0.0056992
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009118, 0.0008671
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100703, 0.0102328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051542, upper bound: 0.0049301
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048208, upper bound: 0.0054753
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032113, 0.0032612
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0047907, 0.0046392
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043748, 0.0042673
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054472, 0.0054849
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051742, 0.0051032
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033716, 0.0033591
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056952, 0.0057039
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009124, 0.0008667
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100610, 0.0102420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051372, upper bound: 0.0050041
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047537, upper bound: 0.0054881
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031683, 0.0031329
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042083, 0.0043074
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035593, 0.0036260
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052447, 0.0052172
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048105, 0.0048487
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032902, 0.0032986
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053222, 0.0053098
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007488, 0.0007660
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089523, 0.0088282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053376, upper bound: 0.0048557
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052673, upper bound: 0.0049300
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031535, 0.0031417
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042401, 0.0042688
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035825, 0.0035974
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052332, 0.0052242
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048168, 0.0048373
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032926, 0.0032953
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053103, 0.0053191
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007488, 0.0007657
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089030, 0.0088628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052295, upper bound: 0.0048887
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051524, upper bound: 0.0049630
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031409, 0.0031235
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039553, 0.0040208
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038348, 0.0038772
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052674, 0.0052550
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047773, 0.0048150
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033116, 0.0033170
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056069, 0.0056362
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008077, 0.0008522
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091930, 0.0091192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053197, upper bound: 0.0049377
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052101, upper bound: 0.0049697
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031395, 0.0031247
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039787, 0.0039982
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038520, 0.0038607
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052671, 0.0052552
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047895, 0.0048020
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033133, 0.0033153
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056261, 0.0056171
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008340, 0.0008240
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091743, 0.0091382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052512, upper bound: 0.0049983
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051366, upper bound: 0.0050340
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030856, 0.0032093
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0043944, 0.0041171
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036807, 0.0035047
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051817, 0.0052758
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048977, 0.0047511
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033067, 0.0032814
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0052987, 0.0053345
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007627, 0.0007510
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0086883, 0.0090786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048616, upper bound: 0.0052656
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047879, upper bound: 0.0053613
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030760, 0.0032185
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0044257, 0.0040868
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037026, 0.0034813
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051738, 0.0052828
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049070, 0.0047418
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033092, 0.0032789
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0052862, 0.0053449
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007632, 0.0007510
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0086517, 0.0091143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048289, upper bound: 0.0053787
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047661, upper bound: 0.0054753
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030626, 0.0031991
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041338, 0.0038385
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0039505, 0.0037597
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052079, 0.0053127
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048642, 0.0047157
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033276, 0.0033005
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055811, 0.0056556
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008221, 0.0008363
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089413, 0.0093618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047815, upper bound: 0.0052845
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047528, upper bound: 0.0053957
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030618, 0.0032006
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041612, 0.0038178
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0039706, 0.0037452
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052077, 0.0053130
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048787, 0.0047062
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033295, 0.0032993
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056056, 0.0056422
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008513, 0.0008085
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089266, 0.0093842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047023, upper bound: 0.0053826
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046841, upper bound: 0.0054881
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032360, 0.0030887
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038485, 0.0042161
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036498, 0.0038931
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052923, 0.0051807
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046690, 0.0048529
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032918, 0.0033240
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056191, 0.0055734
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008152, 0.0008562
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092733, 0.0087868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054881, upper bound: 0.0046841
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054753, upper bound: 0.0047661
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032346, 0.0030896
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038691, 0.0041894
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036642, 0.0038750
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052920, 0.0051808
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046790, 0.0048401
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032930, 0.0033221
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056396, 0.0055507
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008442, 0.0008284
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092518, 0.0088012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053957, upper bound: 0.0047528
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053787, upper bound: 0.0048289
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032178, 0.0030879
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041041, 0.0044182
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0034880, 0.0037008
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052819, 0.0051829
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047534, 0.0049114
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032809, 0.0033084
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053456, 0.0052837
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007404, 0.0007754
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091019, 0.0086846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054881, upper bound: 0.0047537
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053957, upper bound: 0.0048172
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032162, 0.0030897
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041121, 0.0044099
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0034956, 0.0036946
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052806, 0.0051844
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047543, 0.0049109
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032815, 0.0033078
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053409, 0.0052918
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007410, 0.0007750
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090936, 0.0086925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054753, upper bound: 0.0048208
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053787, upper bound: 0.0048802
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032111, 0.0032258
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041053, 0.0041023
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037713, 0.0037779
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052551, 0.0052666
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048232, 0.0048117
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033121, 0.0033108
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055854, 0.0055832
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008442, 0.0008636
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090952, 0.0091233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050340, upper bound: 0.0051366
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050041, upper bound: 0.0051372
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032098, 0.0032270
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041295, 0.0040765
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037872, 0.0037616
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052548, 0.0052669
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048352, 0.0047989
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033137, 0.0033091
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056064, 0.0055663
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008747, 0.0008400
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090764, 0.0091408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049697, upper bound: 0.0052101
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049233, upper bound: 0.0052111
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031420, 0.0031527
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042684, 0.0042402
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035963, 0.0035833
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052245, 0.0052327
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048352, 0.0048178
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032953, 0.0032926
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053214, 0.0053094
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007658, 0.0007492
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088625, 0.0089022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049630, upper bound: 0.0051524
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048887, upper bound: 0.0052295
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031355, 0.0031623
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042838, 0.0042252
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036055, 0.0035748
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052195, 0.0052397
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048441, 0.0048140
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032968, 0.0032913
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053167, 0.0053156
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007547, 0.0007591
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088416, 0.0089267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049301, upper bound: 0.0051542
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048492, upper bound: 0.0052303
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032111, 0.0030829
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041085, 0.0044012
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0034972, 0.0036848
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052770, 0.0051794
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047469, 0.0049000
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032808, 0.0033072
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053391, 0.0052874
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007504, 0.0007632
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090852, 0.0086785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053826, upper bound: 0.0047023
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052845, upper bound: 0.0047815
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031359, 0.0031656
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042999, 0.0042188
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036195, 0.0035667
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052196, 0.0052427
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048457, 0.0048126
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032980, 0.0032910
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053167, 0.0053148
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007662, 0.0007495
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088392, 0.0089434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049983, upper bound: 0.0052512
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0053197
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032572, 0.0032075
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046524, 0.0047638
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042859, 0.0043528
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054813, 0.0054450
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0050954, 0.0051712
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033597, 0.0033697
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056967, 0.0057037
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008767, 0.0008986
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102189, 0.0100643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053613, upper bound: 0.0047879
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049300, upper bound: 0.0052673
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032559, 0.0032085
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046719, 0.0047403
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042966, 0.0043376
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054811, 0.0054452
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051054, 0.0051588
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033610, 0.0033681
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057172, 0.0056798
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009058, 0.0008750
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101993, 0.0100781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052656, upper bound: 0.0048616
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048557, upper bound: 0.0053376
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032538, 0.0032203
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046703, 0.0047625
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042906, 0.0043529
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054786, 0.0054545
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051097, 0.0051693
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033616, 0.0033693
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056961, 0.0057048
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008667, 0.0009113
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102093, 0.0100972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053826, upper bound: 0.0047786
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049807, upper bound: 0.0052541
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032516, 0.0032225
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046791, 0.0047536
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042991, 0.0043457
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054767, 0.0054563
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051106, 0.0051685
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033623, 0.0033686
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056918, 0.0057108
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008674, 0.0009106
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102000, 0.0101065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053613, upper bound: 0.0048569
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049103, upper bound: 0.0052674
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032211, 0.0031143
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039246, 0.0041485
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037018, 0.0038454
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052815, 0.0052001
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047021, 0.0048319
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032981, 0.0033185
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056206, 0.0055694
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008332, 0.0008397
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092000, 0.0088807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052845, upper bound: 0.0048497
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052656, upper bound: 0.0049196
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031434, 0.0031920
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041059, 0.0039638
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038195, 0.0037258
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052218, 0.0052589
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047987, 0.0047386
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033144, 0.0033022
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056003, 0.0055952
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008505, 0.0008241
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089490, 0.0091292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048999, upper bound: 0.0053248
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048286, upper bound: 0.0053382
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032234, 0.0032580
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0047285, 0.0047024
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043281, 0.0043234
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054565, 0.0054801
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051657, 0.0051195
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033671, 0.0033635
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057162, 0.0057110
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009003, 0.0008948
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101206, 0.0101874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053382, upper bound: 0.0048286
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049196, upper bound: 0.0052656
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032225, 0.0032594
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0047536, 0.0046841
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043457, 0.0043117
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054563, 0.0054803
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051796, 0.0051106
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033688, 0.0033623
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057422, 0.0056918
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0008674
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101065, 0.0102070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052674, upper bound: 0.0049103
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048569, upper bound: 0.0053613
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031623, 0.0031436
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042252, 0.0042879
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035748, 0.0036174
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052397, 0.0052232
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048250, 0.0048441
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032915, 0.0032968
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053418, 0.0053167
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007736, 0.0007547
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089267, 0.0088482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052303, upper bound: 0.0048492
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051542, upper bound: 0.0049301
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030897, 0.0032240
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0044099, 0.0041162
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036946, 0.0035068
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051844, 0.0052840
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049211, 0.0047543
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033079, 0.0032815
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053190, 0.0053409
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007893, 0.0007410
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0086925, 0.0091000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048802, upper bound: 0.0053787
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048208, upper bound: 0.0054753
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032203, 0.0030814
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040789, 0.0044369
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0034745, 0.0037191
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052844, 0.0051754
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047511, 0.0049078
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032784, 0.0033098
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053751, 0.0052787
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007651, 0.0007638
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091216, 0.0086483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054881, upper bound: 0.0046841
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053957, upper bound: 0.0047528
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032112, 0.0030914
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041090, 0.0044049
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0034985, 0.0036954
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052772, 0.0051834
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047602, 0.0048986
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032809, 0.0033072
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053646, 0.0052911
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007651, 0.0007634
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090857, 0.0086850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053826, upper bound: 0.0047023
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052845, upper bound: 0.0047815
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031247, 0.0031477
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039982, 0.0039817
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038607, 0.0038622
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052552, 0.0052707
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048133, 0.0047895
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033155, 0.0033133
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056448, 0.0056261
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008393, 0.0008340
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091382, 0.0091803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050340, upper bound: 0.0051366
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049983, upper bound: 0.0052512
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031235, 0.0031490
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040208, 0.0039584
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038772, 0.0038455
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052550, 0.0052710
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048266, 0.0047773
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033171, 0.0033116
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056650, 0.0056069
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008695, 0.0008077
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091192, 0.0091990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049697, upper bound: 0.0052101
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0053197
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031674, 0.0031415
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042084, 0.0043110
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035595, 0.0036367
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052440, 0.0052211
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048238, 0.0048466
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032903, 0.0032986
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053477, 0.0053117
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007634, 0.0007662
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089511, 0.0088347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053376, upper bound: 0.0048557
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052673, upper bound: 0.0049300
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030849, 0.0032179
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0043942, 0.0041207
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036803, 0.0035153
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051811, 0.0052797
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049110, 0.0047479
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033068, 0.0032814
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053242, 0.0053330
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007773, 0.0007505
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0086869, 0.0090852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048616, upper bound: 0.0052656
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047879, upper bound: 0.0053613
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032612, 0.0032190
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046392, 0.0047949
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042673, 0.0043863
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054849, 0.0054507
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051144, 0.0051742
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033592, 0.0033716
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057325, 0.0056952
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008816, 0.0009124
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102420, 0.0100673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054881, upper bound: 0.0047537
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050041, upper bound: 0.0051372
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032598, 0.0032199
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046601, 0.0047687
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042820, 0.0043688
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054846, 0.0054509
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051233, 0.0051601
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033606, 0.0033697
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057535, 0.0056710
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009088, 0.0008852
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102213, 0.0100825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053957, upper bound: 0.0048172
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049233, upper bound: 0.0052111
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0056099, 0.0057677
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0044240, 0.0045345
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053181, 0.0052798
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048756, 0.0049481
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033043, 0.0033175
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056868, 0.0056743
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008567, 0.0008975
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100682, 0.0098788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052295, upper bound: 0.0048887
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053826, upper bound: 0.0047786
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0056290, 0.0057454
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0044374, 0.0045182
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053178, 0.0052800
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048842, 0.0049349
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033057, 0.0033159
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057052, 0.0056473
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008843, 0.0008672
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100500, 0.0098939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051524, upper bound: 0.0049630
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052845, upper bound: 0.0048497
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0057840, 0.0055857
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0045369, 0.0044184
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052586, 0.0053385
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049689, 0.0048485
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033201, 0.0033011
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056626, 0.0056944
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008708, 0.0008809
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0098166, 0.0101219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048289, upper bound: 0.0053787
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049807, upper bound: 0.0052541
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0058103, 0.0055653
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0045551, 0.0044050
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052585, 0.0053388
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049808, 0.0048385
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033220, 0.0032998
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056858, 0.0056720
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009015, 0.0008519
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0098031, 0.0101425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047661, upper bound: 0.0054753
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048999, upper bound: 0.0053248
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0055609, 0.0058147
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043932, 0.0045669
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053348, 0.0052624
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048521, 0.0049672
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033000, 0.0033218
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056992, 0.0056587
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008667, 0.0008867
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101358, 0.0098098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053248, upper bound: 0.0048999
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054753, upper bound: 0.0047661
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0057410, 0.0056334
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0045064, 0.0044491
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052761, 0.0053217
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049486, 0.0048706
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033161, 0.0033055
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056744, 0.0056781
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008820, 0.0008695
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0098872, 0.0100567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048497, upper bound: 0.0052845
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049630, upper bound: 0.0051524
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0055813, 0.0057885
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0044066, 0.0045486
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053346, 0.0052626
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048622, 0.0049552
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033012, 0.0033200
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057215, 0.0056355
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008957, 0.0008559
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101151, 0.0098233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052541, upper bound: 0.0049807
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053787, upper bound: 0.0048289
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0057633, 0.0056143
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0045227, 0.0044358
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052759, 0.0053220
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049618, 0.0048619
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033176, 0.0033042
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057014, 0.0056596
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009124, 0.0008419
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0098721, 0.0100750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047786, upper bound: 0.0053826
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048887, upper bound: 0.0052295
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031608, 0.0031456
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0042353, 0.0042811
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0035819, 0.0036117
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052383, 0.0052249
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048259, 0.0048433
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032922, 0.0032962
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053354, 0.0053215
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007736, 0.0007541
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089183, 0.0088573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052111, upper bound: 0.0049233
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051372, upper bound: 0.0050041
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030879, 0.0032256
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0044182, 0.0041083
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037008, 0.0034994
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051829, 0.0052854
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049216, 0.0047534
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033085, 0.0032809
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0053114, 0.0053456
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007895, 0.0007404
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0086846, 0.0091083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048172, upper bound: 0.0053957
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047537, upper bound: 0.0054881
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032569, 0.0032159
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046534, 0.0047681
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042877, 0.0043640
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054812, 0.0054489
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051089, 0.0051701
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033598, 0.0033697
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057227, 0.0057092
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008917, 0.0009013
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102194, 0.0100710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053613, upper bound: 0.0047879
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049300, upper bound: 0.0052673
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032556, 0.0032168
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046725, 0.0047445
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042979, 0.0043487
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054809, 0.0054491
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051190, 0.0051579
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033611, 0.0033681
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057431, 0.0056837
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009197, 0.0008750
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0101998, 0.0100848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052656, upper bound: 0.0048616
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048557, upper bound: 0.0053376
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032065, 0.0032667
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0047509, 0.0046686
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0043433, 0.0043030
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054434, 0.0054872
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051727, 0.0051043
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033691, 0.0033603
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057021, 0.0057219
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008907, 0.0009056
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100697, 0.0102186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053197, upper bound: 0.0049377
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047815, upper bound: 0.0052845
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032591, 0.0032208
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0046478, 0.0047848
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0042758, 0.0043792
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0054832, 0.0054521
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0051153, 0.0051734
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033598, 0.0033709
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057276, 0.0057017
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008822, 0.0009118
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0102328, 0.0100760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054753, upper bound: 0.0048208
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049301, upper bound: 0.0051542
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0056190, 0.0057600
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0044326, 0.0045290
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053165, 0.0052816
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048765, 0.0049476
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033051, 0.0033169
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056808, 0.0056805
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008574, 0.0008975
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100598, 0.0098881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052101, upper bound: 0.0049697
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053613, upper bound: 0.0048569
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0057920, 0.0055778
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0045431, 0.0044116
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052568, 0.0053398
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049696, 0.0048475
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033207, 0.0033004
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056551, 0.0057008
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008714, 0.0008806
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0098077, 0.0101287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047528, upper bound: 0.0053957
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049103, upper bound: 0.0052674
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0056070, 0.0057761
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0044215, 0.0045403
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053278, 0.0052737
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048767, 0.0049454
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033038, 0.0033185
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057154, 0.0056428
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008842, 0.0008675
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100901, 0.0098653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052512, upper bound: 0.0049983
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053787, upper bound: 0.0048802
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0057787, 0.0055875
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0045320, 0.0044233
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052646, 0.0053290
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049692, 0.0048466
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033191, 0.0033016
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056944, 0.0056656
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009013, 0.0008518
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0098288, 0.0100994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047023, upper bound: 0.0053826
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048492, upper bound: 0.0052303
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0056372, 0.0057375
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0044443, 0.0045125
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0053162, 0.0052818
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048854, 0.0049342
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033063, 0.0033152
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0057000, 0.0056545
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008847, 0.0008673
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0100411, 0.0099022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051366, upper bound: 0.0050340
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052656, upper bound: 0.0049196
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0034802, 0.0034802
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0058171, 0.0055571
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0045603, 0.0043970
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052566, 0.0053401
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0049813, 0.0048375
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033224, 0.0032992
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056795, 0.0056772
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0009021, 0.0008519
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0097933, 0.0101495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046841, upper bound: 0.0054881
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048286, upper bound: 0.0053382
time: 0.81 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053382, upper bound: 0.0048286
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053248, upper bound: 0.0048999
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052674, upper bound: 0.0049103
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052541, upper bound: 0.0049807
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049196, upper bound: 0.0052656
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048569, upper bound: 0.0053613
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048497, upper bound: 0.0052845
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047786, upper bound: 0.0053826
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052303, upper bound: 0.0048492
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052111, upper bound: 0.0049233
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048802, upper bound: 0.0053787
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048172, upper bound: 0.0053957
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0051542, upper bound: 0.0049301
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048208, upper bound: 0.0054753
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0051372, upper bound: 0.0050041
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047537, upper bound: 0.0054881
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053376, upper bound: 0.0048557
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052673, upper bound: 0.0049300
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052295, upper bound: 0.0048887
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0051524, upper bound: 0.0049630
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053197, upper bound: 0.0049377
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052101, upper bound: 0.0049697
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052512, upper bound: 0.0049983
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0051366, upper bound: 0.0050340
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048616, upper bound: 0.0052656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047879, upper bound: 0.0053613
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048289, upper bound: 0.0053787
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047661, upper bound: 0.0054753
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047815, upper bound: 0.0052845
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047528, upper bound: 0.0053957
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047023, upper bound: 0.0053826
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0046841, upper bound: 0.0054881
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0054881, upper bound: 0.0046841
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0054753, upper bound: 0.0047661
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053957, upper bound: 0.0047528
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053787, upper bound: 0.0048289
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0054881, upper bound: 0.0047537
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053957, upper bound: 0.0048172
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0054753, upper bound: 0.0048208
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053787, upper bound: 0.0048802
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0050340, upper bound: 0.0051366
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0050041, upper bound: 0.0051372
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049697, upper bound: 0.0052101
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049233, upper bound: 0.0052111
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049630, upper bound: 0.0051524
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048887, upper bound: 0.0052295
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049301, upper bound: 0.0051542
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048492, upper bound: 0.0052303
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053826, upper bound: 0.0047023
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052845, upper bound: 0.0047815
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049983, upper bound: 0.0052512
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0053197
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053613, upper bound: 0.0047879
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049300, upper bound: 0.0052673
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052656, upper bound: 0.0048616
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048557, upper bound: 0.0053376
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053826, upper bound: 0.0047786
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049807, upper bound: 0.0052541
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053613, upper bound: 0.0048569
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049103, upper bound: 0.0052674
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052845, upper bound: 0.0048497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052656, upper bound: 0.0049196
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048999, upper bound: 0.0053248
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048286, upper bound: 0.0053382
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053382, upper bound: 0.0048286
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049196, upper bound: 0.0052656
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052674, upper bound: 0.0049103
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048569, upper bound: 0.0053613
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052303, upper bound: 0.0048492
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0051542, upper bound: 0.0049301
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048802, upper bound: 0.0053787
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048208, upper bound: 0.0054753
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0054881, upper bound: 0.0046841
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053957, upper bound: 0.0047528
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053826, upper bound: 0.0047023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052845, upper bound: 0.0047815
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0050340, upper bound: 0.0051366
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049983, upper bound: 0.0052512
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049697, upper bound: 0.0052101
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0053197
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053376, upper bound: 0.0048557
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052673, upper bound: 0.0049300
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048616, upper bound: 0.0052656
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047879, upper bound: 0.0053613
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0054881, upper bound: 0.0047537
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0050041, upper bound: 0.0051372
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053957, upper bound: 0.0048172
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049233, upper bound: 0.0052111
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052295, upper bound: 0.0048887
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053826, upper bound: 0.0047786
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0051524, upper bound: 0.0049630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052845, upper bound: 0.0048497
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048289, upper bound: 0.0053787
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049807, upper bound: 0.0052541
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047661, upper bound: 0.0054753
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048999, upper bound: 0.0053248
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053248, upper bound: 0.0048999
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0054753, upper bound: 0.0047661
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048497, upper bound: 0.0052845
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049630, upper bound: 0.0051524
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052541, upper bound: 0.0049807
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053787, upper bound: 0.0048289
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047786, upper bound: 0.0053826
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048887, upper bound: 0.0052295
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052111, upper bound: 0.0049233
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0051372, upper bound: 0.0050041
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048172, upper bound: 0.0053957
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047537, upper bound: 0.0054881
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053613, upper bound: 0.0047879
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049300, upper bound: 0.0052673
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052656, upper bound: 0.0048616
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048557, upper bound: 0.0053376
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053197, upper bound: 0.0049377
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047815, upper bound: 0.0052845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0054753, upper bound: 0.0048208
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049301, upper bound: 0.0051542
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052101, upper bound: 0.0049697
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053613, upper bound: 0.0048569
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047528, upper bound: 0.0053957
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0049103, upper bound: 0.0052674
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052512, upper bound: 0.0049983
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0053787, upper bound: 0.0048802
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0047023, upper bound: 0.0053826
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048492, upper bound: 0.0052303
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0051366, upper bound: 0.0050340
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0052656, upper bound: 0.0049196
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0046841, upper bound: 0.0054881
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 9, lower bound: -0.0048286, upper bound: 0.0053382

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031883, 0.0031365
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039282, 0.0040777
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036901, 0.0037904
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052530, 0.0052138
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047319, 0.0047964
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032989, 0.0033117
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055748, 0.0055708
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008165, 0.0008407
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090923, 0.0089040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052503, upper bound: 0.0039266
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045540, upper bound: 0.0047410
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031868, 0.0031388
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039376, 0.0040709
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036967, 0.0037852
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052518, 0.0052156
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047328, 0.0047957
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032996, 0.0033112
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055697, 0.0055746
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008165, 0.0008401
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090854, 0.0089134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052371, upper bound: 0.0040715
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044557, upper bound: 0.0048087
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031870, 0.0031379
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039532, 0.0040526
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037079, 0.0037732
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052528, 0.0052140
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047458, 0.0047850
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033006, 0.0033099
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055984, 0.0055516
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008429, 0.0008124
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090717, 0.0089236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051815, upper bound: 0.0041744
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042442, upper bound: 0.0048220
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031856, 0.0031402
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039623, 0.0040446
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037150, 0.0037670
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052515, 0.0052159
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047467, 0.0047843
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033013, 0.0033094
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055921, 0.0055561
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008434, 0.0008121
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090648, 0.0089327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051305, upper bound: 0.0048876
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051468, upper bound: 0.0045816
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031111, 0.0032149
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041131, 0.0038978
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038105, 0.0036744
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051947, 0.0052739
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048251, 0.0047013
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033153, 0.0032956
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055522, 0.0055917
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008320, 0.0008249
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088451, 0.0091557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048195, upper bound: 0.0040596
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0044028, upper bound: 0.0051823
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031102, 0.0032163
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041364, 0.0038796
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038291, 0.0036627
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051945, 0.0052742
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048393, 0.0046931
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033171, 0.0032943
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055781, 0.0055772
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008606, 0.0007987
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088311, 0.0091758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039696, upper bound: 0.0040492
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0034847, upper bound: 0.0045358
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031090, 0.0032163
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041190, 0.0038896
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038147, 0.0036675
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051930, 0.0052751
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048256, 0.0047001
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033158, 0.0032950
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055449, 0.0055945
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008320, 0.0008241
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088366, 0.0091633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0040392, upper bound: 0.0039297
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0035916, upper bound: 0.0044234
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031081, 0.0032178
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041429, 0.0038705
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038345, 0.0036541
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051928, 0.0052754
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048400, 0.0046920
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033176, 0.0032936
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055719, 0.0055823
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008609, 0.0007983
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088218, 0.0091837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039643, upper bound: 0.0040492
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0034845, upper bound: 0.0045358
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031738, 0.0031446
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039549, 0.0040393
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037130, 0.0037621
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052418, 0.0052206
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047389, 0.0047839
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033010, 0.0033084
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055632, 0.0055810
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008163, 0.0008401
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090422, 0.0089349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051178, upper bound: 0.0047407
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051270, upper bound: 0.0047522
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031720, 0.0031467
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039638, 0.0040325
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037192, 0.0037565
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052403, 0.0052223
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047399, 0.0047834
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033017, 0.0033078
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055568, 0.0055859
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008164, 0.0008396
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090337, 0.0089446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051109, upper bound: 0.0048169
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051139, upper bound: 0.0048146
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031010, 0.0032250
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041416, 0.0038676
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038329, 0.0036516
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051866, 0.0052814
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048350, 0.0046927
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033177, 0.0032931
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055404, 0.0056041
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008321, 0.0008244
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088074, 0.0091878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047656, upper bound: 0.0052760
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047732, upper bound: 0.0052682
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030991, 0.0032266
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041489, 0.0038597
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038378, 0.0036441
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051850, 0.0052828
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048355, 0.0046916
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033182, 0.0032925
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055329, 0.0056085
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008322, 0.0008237
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087997, 0.0091953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047207, upper bound: 0.0042250
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0041996, upper bound: 0.0053111
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031725, 0.0031460
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039801, 0.0040159
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037297, 0.0037461
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052415, 0.0052209
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047531, 0.0047725
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033029, 0.0033068
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055828, 0.0055611
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008430, 0.0008117
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090236, 0.0089561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0036614, upper bound: 0.0037956
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0036614, upper bound: 0.0037956
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031000, 0.0032265
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041653, 0.0038475
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038503, 0.0036379
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051864, 0.0052818
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048491, 0.0046839
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033193, 0.0032917
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055636, 0.0055877
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008609, 0.0007983
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087936, 0.0092080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047492, upper bound: 0.0053933
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047263, upper bound: 0.0052376
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031708, 0.0031481
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039903, 0.0040080
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037370, 0.0037403
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052401, 0.0052226
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047539, 0.0047718
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033036, 0.0033062
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055769, 0.0055658
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008435, 0.0008114
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090158, 0.0089653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046086, upper bound: 0.0044646
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045215, upper bound: 0.0044954
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030982, 0.0032282
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041736, 0.0038389
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038565, 0.0036295
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051848, 0.0052832
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048498, 0.0046830
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033199, 0.0032911
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055571, 0.0055927
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008612, 0.0007979
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087843, 0.0092162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0045729, upper bound: 0.0053886
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046505, upper bound: 0.0052668
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031792, 0.0031425
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039400, 0.0040624
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036991, 0.0037814
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052464, 0.0052186
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047377, 0.0047892
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033001, 0.0033102
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055692, 0.0055768
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008062, 0.0008517
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090668, 0.0089231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046588, upper bound: 0.0043224
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046600, upper bound: 0.0043203
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031779, 0.0031438
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039634, 0.0040399
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037147, 0.0037649
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052461, 0.0052189
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047498, 0.0047760
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033017, 0.0033085
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055919, 0.0055567
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008325, 0.0008234
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090481, 0.0089427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047033, upper bound: 0.0044637
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045995, upper bound: 0.0044708
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031644, 0.0031513
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039694, 0.0040239
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037219, 0.0037528
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052349, 0.0052256
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047440, 0.0047773
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033025, 0.0033069
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055573, 0.0055866
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008061, 0.0008513
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090175, 0.0089578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0037303, upper bound: 0.0037551
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0037303, upper bound: 0.0037551
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031631, 0.0031526
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039952, 0.0040016
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037380, 0.0037365
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052346, 0.0052259
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047564, 0.0047645
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033042, 0.0033053
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055758, 0.0055660
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008325, 0.0008230
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089992, 0.0089773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049546, upper bound: 0.0048650
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050488, upper bound: 0.0047229
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031778, 0.0031452
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039507, 0.0040555
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037050, 0.0037760
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052452, 0.0052209
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047389, 0.0047883
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033009, 0.0033096
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055641, 0.0055807
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008069, 0.0008514
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090593, 0.0089352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046462, upper bound: 0.0043950
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046486, upper bound: 0.0043950
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031626, 0.0031539
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039787, 0.0040162
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037290, 0.0037474
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052333, 0.0052277
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047452, 0.0047766
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033032, 0.0033063
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055514, 0.0055920
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008069, 0.0008510
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090089, 0.0089684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051104, upper bound: 0.0048621
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051138, upper bound: 0.0048550
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031764, 0.0031464
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039741, 0.0040323
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037222, 0.0037587
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052449, 0.0052211
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047511, 0.0047751
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033026, 0.0033079
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055859, 0.0055615
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008331, 0.0008232
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090399, 0.0089542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051553, upper bound: 0.0049245
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051733, upper bound: 0.0049081
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031613, 0.0031552
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040047, 0.0039936
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037461, 0.0037308
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052330, 0.0052280
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047576, 0.0047636
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033050, 0.0033046
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055705, 0.0055710
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008331, 0.0008229
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089903, 0.0089874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049962, upper bound: 0.0049288
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050319, upper bound: 0.0049280
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030961, 0.0032189
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041235, 0.0038722
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038168, 0.0036601
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051834, 0.0052771
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048249, 0.0046877
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033164, 0.0032930
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055456, 0.0055966
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008201, 0.0008361
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088028, 0.0091705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046449, upper bound: 0.0051642
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047612, upper bound: 0.0050082
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030952, 0.0032204
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041495, 0.0038531
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038362, 0.0036499
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051832, 0.0052775
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048393, 0.0046784
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033183, 0.0032917
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055711, 0.0055815
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008491, 0.0008084
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087892, 0.0091931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046741, upper bound: 0.0052619
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046799, upper bound: 0.0052623
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030864, 0.0032281
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041540, 0.0038419
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038395, 0.0036367
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051755, 0.0052841
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048342, 0.0046784
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033189, 0.0032905
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055332, 0.0056088
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008206, 0.0008361
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087661, 0.0092071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043888, upper bound: 0.0046213
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043858, upper bound: 0.0047588
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030856, 0.0032296
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041808, 0.0038215
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038580, 0.0036233
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051754, 0.0052845
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048479, 0.0046691
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033208, 0.0032892
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055563, 0.0055918
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008498, 0.0008083
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087526, 0.0092287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046742, upper bound: 0.0044915
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0040352, upper bound: 0.0053895
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030941, 0.0032208
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041292, 0.0038641
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038207, 0.0036540
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051817, 0.0052786
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048258, 0.0046865
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033169, 0.0032923
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055381, 0.0056001
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008208, 0.0008354
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087942, 0.0091778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046919, upper bound: 0.0042101
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0041454, upper bound: 0.0051996
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030844, 0.0032299
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041612, 0.0038339
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038446, 0.0036299
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051738, 0.0052857
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048350, 0.0046773
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033195, 0.0032899
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055256, 0.0056140
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008212, 0.0008354
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087573, 0.0092139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046583, upper bound: 0.0053178
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046767, upper bound: 0.0052623
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030933, 0.0032223
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041566, 0.0038437
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038408, 0.0036417
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051815, 0.0052789
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048403, 0.0046772
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033188, 0.0032910
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055649, 0.0055866
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008497, 0.0008077
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087785, 0.0092001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046784, upper bound: 0.0053509
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046727, upper bound: 0.0053509
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0030835, 0.0032315
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041886, 0.0038133
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038643, 0.0036154
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051736, 0.0052860
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048489, 0.0046678
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033214, 0.0032886
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055500, 0.0055966
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008505, 0.0008077
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0087426, 0.0092361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0041696, upper bound: 0.0047767
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0041708, upper bound: 0.0047766
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032316, 0.0030825
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038129, 0.0041883
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036143, 0.0038639
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052860, 0.0051729
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046650, 0.0048498
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032886, 0.0033214
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055965, 0.0055460
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008079, 0.0008490
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092358, 0.0087418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053837, upper bound: 0.0045875
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053841, upper bound: 0.0045875
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032298, 0.0030845
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038210, 0.0041805
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036221, 0.0038577
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052846, 0.0051747
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046660, 0.0048489
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032892, 0.0033208
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055917, 0.0055525
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008080, 0.0008480
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092283, 0.0087516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052599, upper bound: 0.0046908
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053933, upper bound: 0.0046930
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032301, 0.0030834
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038334, 0.0041608
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036287, 0.0038446
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052856, 0.0051731
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046750, 0.0048369
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032898, 0.0033194
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056175, 0.0055233
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008367, 0.0008212
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092135, 0.0087562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053111, upper bound: 0.0041131
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042947, upper bound: 0.0046635
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032284, 0.0030854
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038413, 0.0041538
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036359, 0.0038395
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052842, 0.0051749
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046761, 0.0048361
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032905, 0.0033189
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056121, 0.0055312
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008369, 0.0008206
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092068, 0.0087650

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
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047167, upper bound: 0.0042748
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047174, upper bound: 0.0042748
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032289, 0.0030975
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038386, 0.0041733
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036287, 0.0038563
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052836, 0.0051844
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046807, 0.0048519
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032911, 0.0033199
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055925, 0.0055537
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007978, 0.0008601
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092164, 0.0087839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049391, upper bound: 0.0042411
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049034, upper bound: 0.0042578
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032274, 0.0030985
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038592, 0.0041486
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036434, 0.0038378
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052833, 0.0051846
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046896, 0.0048386
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032925, 0.0033182
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056121, 0.0055306
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008250, 0.0008328
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091952, 0.0087990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053431, upper bound: 0.0047205
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053516, upper bound: 0.0047950
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032272, 0.0030993
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038470, 0.0041650
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036376, 0.0038501
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052823, 0.0051858
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046816, 0.0048511
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032917, 0.0033193
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055878, 0.0055604
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007984, 0.0008594
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0092080, 0.0087926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047586, upper bound: 0.0042737
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0047630, upper bound: 0.0042737
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032258, 0.0031003
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038672, 0.0041414
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036510, 0.0038328
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052819, 0.0051861
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046906, 0.0048381
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032930, 0.0033177
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056075, 0.0055387
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008255, 0.0008324
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091878, 0.0088070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052324, upper bound: 0.0048068
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052998, upper bound: 0.0048051
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031556, 0.0031605
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039933, 0.0040047
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037301, 0.0037463
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052282, 0.0052325
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047617, 0.0047591
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033046, 0.0033050
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055734, 0.0055662
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008233, 0.0008309
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089873, 0.0089895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0049375, upper bound: 0.0040124
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0044443, upper bound: 0.0050524
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031486, 0.0031703
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040077, 0.0039904
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037397, 0.0037373
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052229, 0.0052398
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047706, 0.0047559
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033062, 0.0033036
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055684, 0.0055726
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008114, 0.0008425
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089656, 0.0090153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048997, upper bound: 0.0050319
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048994, upper bound: 0.0049987
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031543, 0.0031619
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040158, 0.0039790
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037462, 0.0037299
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052280, 0.0052328
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047751, 0.0047463
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033063, 0.0033032
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055954, 0.0055493
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008535, 0.0008072
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089685, 0.0090082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049432, upper bound: 0.0051829
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049263, upper bound: 0.0051846
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031472, 0.0031716
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040320, 0.0039641
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037555, 0.0037199
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052226, 0.0052400
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047826, 0.0047421
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033078, 0.0033017
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055894, 0.0055551
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008420, 0.0008170
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089451, 0.0090328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0037551, upper bound: 0.0037303
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0037551, upper bound: 0.0037303
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031529, 0.0031623
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040012, 0.0039953
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037356, 0.0037387
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052261, 0.0052341
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047624, 0.0047579
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033053, 0.0033042
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055683, 0.0055718
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008232, 0.0008300
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089770, 0.0089985

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
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043903, upper bound: 0.0045533
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043936, upper bound: 0.0045533
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031516, 0.0031636
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040235, 0.0039694
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037517, 0.0037230
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052259, 0.0052343
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047757, 0.0047451
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033069, 0.0033025
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055899, 0.0055563
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008536, 0.0008065
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089577, 0.0090167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0041010, upper bound: 0.0039253
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0036454, upper bound: 0.0044005
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031465, 0.0031719
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040155, 0.0039803
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037452, 0.0037302
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052212, 0.0052411
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047714, 0.0047551
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033067, 0.0033029
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055636, 0.0055791
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008121, 0.0008418
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089561, 0.0090229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048409, upper bound: 0.0041647
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0041994, upper bound: 0.0050689
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031451, 0.0031731
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040389, 0.0039551
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037610, 0.0037138
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052209, 0.0052413
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047831, 0.0047413
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033084, 0.0033010
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055845, 0.0055625
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008425, 0.0008165
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089351, 0.0090411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043053, upper bound: 0.0045972
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043211, upper bound: 0.0045972
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032221, 0.0030925
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038430, 0.0041563
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036399, 0.0038402
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052787, 0.0051809
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046741, 0.0048402
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032909, 0.0033188
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055860, 0.0055598
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008078, 0.0008483
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091996, 0.0087773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052179, upper bound: 0.0046243
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053058, upper bound: 0.0046243
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032207, 0.0030934
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038635, 0.0041288
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036526, 0.0038204
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052784, 0.0051811
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046841, 0.0048272
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032923, 0.0033169
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056028, 0.0055343
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008368, 0.0008206
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091771, 0.0087930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051872, upper bound: 0.0047045
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052079, upper bound: 0.0046708
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031468, 0.0031752
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040320, 0.0039739
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037579, 0.0037221
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052213, 0.0052441
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047730, 0.0047522
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033079, 0.0033026
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055636, 0.0055819
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008236, 0.0008307
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089537, 0.0090386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0046554, upper bound: 0.0051438
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049033, upper bound: 0.0051277
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031455, 0.0031766
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040550, 0.0039506
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037749, 0.0037054
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052210, 0.0052444
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047865, 0.0047399
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033096, 0.0033009
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055838, 0.0055617
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008538, 0.0008069
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089350, 0.0090579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0037311, upper bound: 0.0037710
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0037311, upper bound: 0.0037710
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032205, 0.0030944
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038521, 0.0041492
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036481, 0.0038357
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052773, 0.0051826
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046752, 0.0048392
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032917, 0.0033182
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055799, 0.0055656
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008079, 0.0008474
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091926, 0.0087876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052623, upper bound: 0.0046799
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052619, upper bound: 0.0046741
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031441, 0.0031770
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040395, 0.0039635
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037642, 0.0037150
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052190, 0.0052454
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047738, 0.0047510
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033085, 0.0033017
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055586, 0.0055879
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008236, 0.0008298
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089422, 0.0090467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048372, upper bound: 0.0041825
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0042417, upper bound: 0.0051815
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032190, 0.0030954
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038716, 0.0041232
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036587, 0.0038164
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052770, 0.0051828
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046852, 0.0048263
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032930, 0.0033164
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055983, 0.0055417
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008370, 0.0008200
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091699, 0.0088014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050143, upper bound: 0.0047612
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051643, upper bound: 0.0046197
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031429, 0.0031783
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040619, 0.0039399
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037803, 0.0036997
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052187, 0.0052457
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047870, 0.0047385
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033102, 0.0033001
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055791, 0.0055671
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008540, 0.0008062
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089226, 0.0090656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0037311, upper bound: 0.0037710
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0037311, upper bound: 0.0037710
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032179, 0.0031072
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038700, 0.0041425
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036528, 0.0038340
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052753, 0.0051922
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046894, 0.0048416
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032936, 0.0033176
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055813, 0.0055667
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007979, 0.0008597
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091831, 0.0088205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052063, upper bound: 0.0046927
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053058, upper bound: 0.0047043
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031407, 0.0031846
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040443, 0.0039622
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037660, 0.0037151
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052162, 0.0052509
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047827, 0.0047491
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033093, 0.0033013
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055580, 0.0055878
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008120, 0.0008425
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089326, 0.0090636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043953, upper bound: 0.0046199
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0043972, upper bound: 0.0046199
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032163, 0.0031094
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038788, 0.0041360
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036613, 0.0038285
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052741, 0.0051940
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046904, 0.0048409
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032943, 0.0033171
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055760, 0.0055727
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0007986, 0.0008591
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091754, 0.0088298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048559, upper bound: 0.0043156
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048145, upper bound: 0.0043220
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031385, 0.0031861
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040523, 0.0039532
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037724, 0.0037079
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052144, 0.0052522
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047835, 0.0047483
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033099, 0.0033006
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055537, 0.0055943
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008127, 0.0008418
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089233, 0.0090704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048152, upper bound: 0.0051599
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047917, upper bound: 0.0051373
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032164, 0.0031081
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038890, 0.0041188
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036663, 0.0038142
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052750, 0.0051924
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046981, 0.0048284
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032949, 0.0033158
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055974, 0.0055419
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008255, 0.0008325
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091627, 0.0088357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050133, upper bound: 0.0047462
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051856, upper bound: 0.0046836
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0032149, 0.0031103
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0038973, 0.0041128
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036731, 0.0038100
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052738, 0.0051941
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0046993, 0.0048279
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032956, 0.0033153
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055931, 0.0055490
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008259, 0.0008320
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0091550, 0.0088439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046779, upper bound: 0.0044727
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0045069, upper bound: 0.0044727
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031394, 0.0031858
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040703, 0.0039375
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037840, 0.0036969
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052159, 0.0052511
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047947, 0.0047355
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033112, 0.0032996
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055776, 0.0055677
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008428, 0.0008169
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089135, 0.0090842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 180

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048087, upper bound: 0.0044557
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0041048, upper bound: 0.0052371
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031372, 0.0031873
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0040772, 0.0039282
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037893, 0.0036903
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052141, 0.0052524
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047952, 0.0047346
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033117, 0.0032989
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055729, 0.0055735
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008433, 0.0008164
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0089040, 0.0090913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047362, upper bound: 0.0052603
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047515, upper bound: 0.0052103
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031873, 0.0031451
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039282, 0.0040816
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0036903, 0.0038016
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052524, 0.0052177
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047452, 0.0047952
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0032991, 0.0033117
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056013, 0.0055729
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008317, 0.0008433
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090913, 0.0089106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052103, upper bound: 0.0047515
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052603, upper bound: 0.0047362
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031103, 0.0032234
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041128, 0.0039017
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038100, 0.0036857
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051941, 0.0052779
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048383, 0.0046993
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033155, 0.0032956
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0055786, 0.0055931
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008471, 0.0008259
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088439, 0.0091623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 13
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0047140, upper bound: 0.0051642
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0048170, upper bound: 0.0050133
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031861, 0.0031464
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0039532, 0.0040565
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0037079, 0.0037845
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0052522, 0.0052180
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0047591, 0.0047835
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033008, 0.0033099
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056249, 0.0055537
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008580, 0.0008127
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0090704, 0.0089301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 180
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 183
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046297, upper bound: 0.0043421
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0046297, upper bound: 0.0043260
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0038589, -0.0003786, -0.0038589, -0.0003786, -0.0031094, 0.0032249
1: -0.0043433, 0.0041499, -0.0043433, 0.0041499, -0.0041360, 0.0038834
2: 0.0030233, 0.0101079, 0.0030233, 0.0101079, -0.0038285, 0.0036740
3: -0.0045443, -0.0035018, -0.0045443, -0.0035018, -0.0010425, 0.0010425
4: 0.0015544, 0.0078154, 0.0015544, 0.0078154, -0.0051940, 0.0052782
5: -0.0029806, 0.0033723, -0.0029806, 0.0033723, -0.0048526, 0.0046904
6: -0.0068059, -0.0032665, -0.0068059, -0.0032665, -0.0033172, 0.0032943
7: -0.0022910, 0.0037843, -0.0022910, 0.0037843, -0.0056046, 0.0055760
8: -0.0008404, 0.0000793, -0.0008404, 0.0000793, -0.0008758, 0.0007986
9: 0.9969709, 1.0119554, 0.9969709, 1.0119554, -0.0088298, 0.0091824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.03 + 598.24 = 601.27 seconds
