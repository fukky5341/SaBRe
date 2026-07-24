## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00017731


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001604, 0.0001604)
1: (-0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0004071, 0.0004071)
2: (0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0002525, 0.0002525)
3: (0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0004716, 0.0004716)
4: (-0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0004140, 0.0004140)
5: (0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001568, 0.0001568)
6: (0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0005985, 0.0005985)
7: (0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0004188, 0.0004188)
8: (-0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0004490, 0.0004490)
9: (-0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002966, 0.0002966)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 1.35 = 2.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0002483, upper bound: 0.0002483

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002348, upper bound: 0.0002289
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002289, upper bound: 0.0002349
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.08 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 7, lower bound: -0.0002348, upper bound: 0.0002289
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 7, lower bound: -0.0002289, upper bound: 0.0002349

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001295, 0.0001316
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0003287, 0.0003339
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0002039, 0.0002071
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003868, 0.0003808
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0003344, 0.0003396
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001266, 0.0001286
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004908, 0.0004833
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0003435, 0.0003382
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003683, 0.0003626
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002395, 0.0002433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002230, upper bound: 0.0002179
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002234, upper bound: 0.0002172
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001316, 0.0001295
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0003339, 0.0003287
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0002071, 0.0002039
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003808, 0.0003868
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0003396, 0.0003344
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001286, 0.0001266
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004833, 0.0004908
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0003382, 0.0003435
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003626, 0.0003683
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002433, 0.0002395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002174, upper bound: 0.0002238
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002179, upper bound: 0.0002235
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 7, lower bound: -0.0002230, upper bound: 0.0002179
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 7, lower bound: -0.0002234, upper bound: 0.0002172
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 7, lower bound: -0.0002174, upper bound: 0.0002238
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 7, lower bound: -0.0002179, upper bound: 0.0002235

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001216, 0.0001237
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0003086, 0.0003138
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001915, 0.0001947
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003635, 0.0003575
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0003139, 0.0003192
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001189, 0.0001209
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004614, 0.0004538
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0003228, 0.0003175
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003461, 0.0003404
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002249, 0.0002286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002115, upper bound: 0.0002065
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002118, upper bound: 0.0002060
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001216, 0.0001237
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0003087, 0.0003139
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001915, 0.0001947
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003636, 0.0003576
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0003140, 0.0003193
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001189, 0.0001209
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004615, 0.0004538
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0003229, 0.0003176
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003462, 0.0003405
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002249, 0.0002287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002117
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002177, upper bound: 0.0002074
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001215, 0.0001196
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0003083, 0.0003036
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001913, 0.0001884
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003517, 0.0003571
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0003136, 0.0003088
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001188, 0.0001170
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004464, 0.0004532
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0003124, 0.0003172
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003349, 0.0003400
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002246, 0.0002212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002053, upper bound: 0.0002120
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002060, upper bound: 0.0002118
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001217, 0.0001189
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0003088, 0.0003018
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001916, 0.0001873
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003497, 0.0003577
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0003141, 0.0003070
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001190, 0.0001163
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004438, 0.0004539
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0003105, 0.0003176
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003329, 0.0003406
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002250, 0.0002199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002057, upper bound: 0.0002117
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002064, upper bound: 0.0002115
time: 0.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0002115, upper bound: 0.0002065
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0002118, upper bound: 0.0002060
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0002141, upper bound: 0.0002117
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0002177, upper bound: 0.0002074
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0002053, upper bound: 0.0002120
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0002060, upper bound: 0.0002118
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0002057, upper bound: 0.0002117
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 7, lower bound: -0.0002064, upper bound: 0.0002115

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001107, 0.0001133
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002809, 0.0002875
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001742, 0.0001784
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003331, 0.0003254
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002857, 0.0002925
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001082, 0.0001108
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004228, 0.0004129
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002958, 0.0002889
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003172, 0.0003098
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002046, 0.0002095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002015, upper bound: 0.0002012
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002058, upper bound: 0.0001951
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001113, 0.0001131
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002824, 0.0002871
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001752, 0.0001781
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003326, 0.0003271
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002872, 0.0002920
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001088, 0.0001106
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004220, 0.0004152
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002953, 0.0002905
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003166, 0.0003115
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002057, 0.0002092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002015, upper bound: 0.0002007
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002060, upper bound: 0.0001951
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001128, 0.0001135
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002863, 0.0002881
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001776, 0.0001787
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003338, 0.0003317
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002912, 0.0002931
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001103, 0.0001110
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004236, 0.0004209
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002964, 0.0002946
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003178, 0.0003158
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002086, 0.0002099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002015, upper bound: 0.0002004
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002015, upper bound: 0.0001999
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001115, 0.0001149
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002829, 0.0002916
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001755, 0.0001809
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003378, 0.0003277
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002878, 0.0002966
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001090, 0.0001123
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004287, 0.0004159
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0003000, 0.0002910
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003216, 0.0003120
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002061, 0.0002125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002062, upper bound: 0.0001949
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002065, upper bound: 0.0001949
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001129, 0.0001113
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002864, 0.0002824
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001777, 0.0001752
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003272, 0.0003318
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002913, 0.0002873
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001103, 0.0001088
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004152, 0.0004211
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002905, 0.0002946
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003115, 0.0003159
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002087, 0.0002058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001949, upper bound: 0.0002065
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001999, upper bound: 0.0002015
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001131, 0.0001113
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002871, 0.0002824
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001781, 0.0001752
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003271, 0.0003326
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002920, 0.0002872
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001106, 0.0001088
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004152, 0.0004220
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002905, 0.0002953
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003115, 0.0003166
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002092, 0.0002057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001951, upper bound: 0.0002060
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002007, upper bound: 0.0002015
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001133, 0.0001106
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002876, 0.0002806
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001784, 0.0001741
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003251, 0.0003332
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002926, 0.0002854
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001108, 0.0001081
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004126, 0.0004229
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002887, 0.0002959
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003095, 0.0003173
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002096, 0.0002045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001949, upper bound: 0.0002062
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002004, upper bound: 0.0002015
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001133, 0.0001107
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002875, 0.0002809
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001784, 0.0001742
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003254, 0.0003331
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002925, 0.0002857
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001108, 0.0001082
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0004129, 0.0004228
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002889, 0.0002958
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0003098, 0.0003172
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0002095, 0.0002046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001951, upper bound: 0.0002058
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002011, upper bound: 0.0002015
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002015, upper bound: 0.0002012
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002058, upper bound: 0.0001951
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002015, upper bound: 0.0002007
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002060, upper bound: 0.0001951
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002015, upper bound: 0.0002004
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002015, upper bound: 0.0001999
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002062, upper bound: 0.0001949
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002065, upper bound: 0.0001949
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0001949, upper bound: 0.0002065
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0001999, upper bound: 0.0002015
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0001951, upper bound: 0.0002060
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002007, upper bound: 0.0002015
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0001949, upper bound: 0.0002062
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002004, upper bound: 0.0002015
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0001951, upper bound: 0.0002058
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 7, lower bound: -0.0002011, upper bound: 0.0002015

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001029, 0.0001037
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002612, 0.0002633
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001621, 0.0001633
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003050, 0.0003026
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002657, 0.0002678
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001006, 0.0001014
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003870, 0.0003840
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002708, 0.0002687
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002904, 0.0002881
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001903, 0.0001918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001810, upper bound: 0.0001809
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001815, upper bound: 0.0001806
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001011, 0.0001051
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002566, 0.0002666
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001592, 0.0001654
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003089, 0.0002972
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002610, 0.0002712
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000989, 0.0001027
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003920, 0.0003772
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002743, 0.0002640
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002941, 0.0002830
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001869, 0.0001943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002051, upper bound: 0.0001788
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001880, upper bound: 0.0001944
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001034, 0.0001036
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002623, 0.0002628
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001628, 0.0001630
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003044, 0.0003039
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002668, 0.0002673
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001011, 0.0001012
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003863, 0.0003857
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002703, 0.0002699
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002898, 0.0002894
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001911, 0.0001915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001826, upper bound: 0.0001740
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001788, upper bound: 0.0001812
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001017, 0.0001049
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002581, 0.0002663
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001601, 0.0001652
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003085, 0.0002990
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002625, 0.0002709
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000994, 0.0001026
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003915, 0.0003795
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002739, 0.0002655
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002937, 0.0002847
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001881, 0.0001940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002052, upper bound: 0.0001793
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001880, upper bound: 0.0001944
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001025, 0.0001038
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002601, 0.0002633
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001614, 0.0001634
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003051, 0.0003014
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002646, 0.0002678
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001002, 0.0001015
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003872, 0.0003825
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002709, 0.0002676
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002905, 0.0002870
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001895, 0.0001919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001827, upper bound: 0.0001736
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001794, upper bound: 0.0001807
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001031, 0.0001033
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002615, 0.0002621
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001623, 0.0001626
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003036, 0.0003030
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002660, 0.0002666
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001008, 0.0001010
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003853, 0.0003845
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002696, 0.0002691
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002891, 0.0002885
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001906, 0.0001910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002008, upper bound: 0.0001814
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001864, upper bound: 0.0001991
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001010, 0.0001051
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002563, 0.0002668
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001590, 0.0001655
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003091, 0.0002969
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002607, 0.0002714
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000988, 0.0001028
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003923, 0.0003769
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002745, 0.0002637
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002943, 0.0002827
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001868, 0.0001944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002055, upper bound: 0.0001787
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001879, upper bound: 0.0001941
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001017, 0.0001048
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002581, 0.0002660
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001601, 0.0001650
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003081, 0.0002990
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002626, 0.0002705
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000994, 0.0001025
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003910, 0.0003795
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002736, 0.0002656
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002934, 0.0002847
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001881, 0.0001938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001868, upper bound: 0.0001732
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001807, upper bound: 0.0001760
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001048, 0.0001017
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002660, 0.0002581
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001650, 0.0001601
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002990, 0.0003081
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002705, 0.0002626
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001025, 0.0000994
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003795, 0.0003910
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002656, 0.0002736
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002847, 0.0002934
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001938, 0.0001881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001941, upper bound: 0.0001879
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001792, upper bound: 0.0002058
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001033, 0.0001031
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002621, 0.0002615
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001626, 0.0001623
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003030, 0.0003036
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002666, 0.0002660
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001010, 0.0001008
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003845, 0.0003853
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002691, 0.0002696
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002885, 0.0002891
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001910, 0.0001906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001992, upper bound: 0.0001864
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001814, upper bound: 0.0002008
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001049, 0.0001017
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002663, 0.0002581
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001652, 0.0001601
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002990, 0.0003085
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002709, 0.0002625
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001026, 0.0000994
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003795, 0.0003915
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002655, 0.0002739
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002847, 0.0002937
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001940, 0.0001881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001745, upper bound: 0.0001851
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001759, upper bound: 0.0001851
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001036, 0.0001034
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002628, 0.0002623
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001630, 0.0001628
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003039, 0.0003044
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002673, 0.0002668
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001012, 0.0001011
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003857, 0.0003863
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002699, 0.0002703
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002894, 0.0002898
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001915, 0.0001911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001803, upper bound: 0.0001815
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001809, upper bound: 0.0001810
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001051, 0.0001010
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002668, 0.0002563
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001655, 0.0001590
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002969, 0.0003091
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002714, 0.0002607
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001028, 0.0000988
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003769, 0.0003923
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002637, 0.0002745
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002827, 0.0002943
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001944, 0.0001868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001901, upper bound: 0.0001962
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001880, upper bound: 0.0002015
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001038, 0.0001025
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002633, 0.0002601
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001634, 0.0001614
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003014, 0.0003051
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002678, 0.0002646
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001015, 0.0001002
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003825, 0.0003872
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002676, 0.0002709
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002870, 0.0002905
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001919, 0.0001895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001996, upper bound: 0.0001864
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001810, upper bound: 0.0002008
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001051, 0.0001011
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002666, 0.0002566
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001654, 0.0001592
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002972, 0.0003089
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002712, 0.0002610
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001027, 0.0000989
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003772, 0.0003920
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002640, 0.0002743
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002830, 0.0002941
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001943, 0.0001869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001904, upper bound: 0.0001963
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001885, upper bound: 0.0002010
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001037, 0.0001029
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002633, 0.0002612
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001633, 0.0001621
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003026, 0.0003050
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002678, 0.0002657
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001014, 0.0001006
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003840, 0.0003870
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002687, 0.0002708
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002881, 0.0002904
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001918, 0.0001903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002004, upper bound: 0.0001865
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001811, upper bound: 0.0002007
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001810, upper bound: 0.0001809
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001815, upper bound: 0.0001806
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0002051, upper bound: 0.0001788
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001880, upper bound: 0.0001944
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001826, upper bound: 0.0001740
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001788, upper bound: 0.0001812
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0002052, upper bound: 0.0001793
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001880, upper bound: 0.0001944
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001827, upper bound: 0.0001736
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001794, upper bound: 0.0001807
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0002008, upper bound: 0.0001814
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001864, upper bound: 0.0001991
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0002055, upper bound: 0.0001787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001879, upper bound: 0.0001941
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001868, upper bound: 0.0001732
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001807, upper bound: 0.0001760
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001941, upper bound: 0.0001879
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001792, upper bound: 0.0002058
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001992, upper bound: 0.0001864
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001814, upper bound: 0.0002008
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001745, upper bound: 0.0001851
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001759, upper bound: 0.0001851
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001803, upper bound: 0.0001815
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001809, upper bound: 0.0001810
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001901, upper bound: 0.0001962
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001880, upper bound: 0.0002015
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001996, upper bound: 0.0001864
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001810, upper bound: 0.0002008
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001904, upper bound: 0.0001963
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001885, upper bound: 0.0002010
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0002004, upper bound: 0.0001865
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 7, lower bound: -0.0001811, upper bound: 0.0002007

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000937, 0.0000947
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002379, 0.0002403
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001476, 0.0001491
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002784, 0.0002756
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002419, 0.0002444
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000916, 0.0000926
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003533, 0.0003497
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002472, 0.0002447
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002650, 0.0002624
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001733, 0.0001751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001548
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001588, upper bound: 0.0001617
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000939, 0.0000948
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002382, 0.0002406
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001478, 0.0001493
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002788, 0.0002760
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002423, 0.0002448
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000918, 0.0000927
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003538, 0.0003503
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002476, 0.0002451
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002654, 0.0002628
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001736, 0.0001753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001769, upper bound: 0.0001731
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001728, upper bound: 0.0001762
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001012, 0.0001054
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002568, 0.0002676
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001593, 0.0001660
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003100, 0.0002975
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002612, 0.0002722
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000990, 0.0001031
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003934, 0.0003776
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002753, 0.0002642
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002951, 0.0002833
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001871, 0.0001950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0002002, upper bound: 0.0001699
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001955, upper bound: 0.0001736
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001015, 0.0001052
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002575, 0.0002669
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001598, 0.0001656
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003092, 0.0002983
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002619, 0.0002715
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000992, 0.0001028
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003924, 0.0003786
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002746, 0.0002649
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002944, 0.0002841
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001876, 0.0001945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001637, upper bound: 0.0001750
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001602, upper bound: 0.0001738
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000830, 0.0000854
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002107, 0.0002166
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001307, 0.0001344
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002509, 0.0002441
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002143, 0.0002203
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000812, 0.0000834
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003184, 0.0003098
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002228, 0.0002168
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002389, 0.0002324
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001535, 0.0001578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001774, upper bound: 0.0001663
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001746, upper bound: 0.0001692
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000852, 0.0000837
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002161, 0.0002124
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001341, 0.0001318
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002461, 0.0002504
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002199, 0.0002161
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000833, 0.0000818
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003123, 0.0003178
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002185, 0.0002224
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002343, 0.0002384
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001575, 0.0001548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001737, upper bound: 0.0001720
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001707, upper bound: 0.0001761
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001018, 0.0001053
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002584, 0.0002673
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001603, 0.0001658
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003097, 0.0002993
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002628, 0.0002719
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000995, 0.0001030
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003930, 0.0003798
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002750, 0.0002658
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002948, 0.0002850
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001882, 0.0001948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001858, upper bound: 0.0001588
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001801, upper bound: 0.0001607
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001021, 0.0001050
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002591, 0.0002665
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001608, 0.0001654
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003088, 0.0003002
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002636, 0.0002711
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000998, 0.0001027
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003919, 0.0003809
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002742, 0.0002666
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002940, 0.0002858
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001888, 0.0001942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001631, upper bound: 0.0001751
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001556, upper bound: 0.0001737
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000822, 0.0000856
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002085, 0.0002171
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001293, 0.0001347
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002516, 0.0002415
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002121, 0.0002209
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000803, 0.0000837
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003193, 0.0003065
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002234, 0.0002145
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002395, 0.0002300
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001519, 0.0001582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001775, upper bound: 0.0001663
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001734, upper bound: 0.0001685
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000843, 0.0000841
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002140, 0.0002133
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001327, 0.0001323
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002471, 0.0002479
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002176, 0.0002170
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000824, 0.0000822
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003136, 0.0003146
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002194, 0.0002201
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002353, 0.0002360
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001559, 0.0001554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001740
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001704, upper bound: 0.0001758
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001032, 0.0001037
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002618, 0.0002630
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001624, 0.0001632
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003047, 0.0003033
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002663, 0.0002675
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001009, 0.0001013
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003867, 0.0003849
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002706, 0.0002693
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002901, 0.0002888
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001907, 0.0001916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001803, upper bound: 0.0001559
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001808, upper bound: 0.0001579
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001035, 0.0001034
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002626, 0.0002624
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001629, 0.0001628
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003039, 0.0003042
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002671, 0.0002669
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001012, 0.0001011
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003857, 0.0003861
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002699, 0.0002702
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002894, 0.0002897
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001913, 0.0001912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001809, upper bound: 0.0001904
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001736, upper bound: 0.0001945
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001011, 0.0001056
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002566, 0.0002679
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001592, 0.0001662
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003104, 0.0002972
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002610, 0.0002725
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000989, 0.0001032
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003939, 0.0003772
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002756, 0.0002640
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002955, 0.0002830
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001870, 0.0001952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001860, upper bound: 0.0001577
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001805, upper bound: 0.0001601
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001014, 0.0001052
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002573, 0.0002671
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001596, 0.0001657
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003094, 0.0002981
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002617, 0.0002717
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000991, 0.0001029
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003927, 0.0003783
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002748, 0.0002647
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002946, 0.0002838
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001875, 0.0001946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001688, upper bound: 0.0001721
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001663, upper bound: 0.0001754
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000818, 0.0000866
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002076, 0.0002198
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001288, 0.0001363
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002546, 0.0002405
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002112, 0.0002235
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000800, 0.0000847
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003231, 0.0003053
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002261, 0.0002136
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002424, 0.0002290
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001513, 0.0001601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001862, upper bound: 0.0001589
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001689, upper bound: 0.0001726
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000835, 0.0000844
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002119, 0.0002142
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001315, 0.0001329
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002482, 0.0002455
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002156, 0.0002179
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000817, 0.0000825
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003150, 0.0003116
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002204, 0.0002180
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002363, 0.0002338
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001544, 0.0001561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001801, upper bound: 0.0001607
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001657, upper bound: 0.0001754
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001049, 0.0001021
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002662, 0.0002591
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001652, 0.0001607
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003001, 0.0003084
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002708, 0.0002635
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001026, 0.0000998
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003809, 0.0003914
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002665, 0.0002739
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002858, 0.0002936
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001940, 0.0001888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001754, upper bound: 0.0001657
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001726, upper bound: 0.0001689
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001052, 0.0001018
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002670, 0.0002584
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001656, 0.0001603
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002993, 0.0003093
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002716, 0.0002628
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001029, 0.0000995
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003799, 0.0003925
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002658, 0.0002747
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002850, 0.0002945
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001945, 0.0001883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001607, upper bound: 0.0001801
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001588, upper bound: 0.0001862
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001034, 0.0001035
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002624, 0.0002626
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001628, 0.0001629
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003042, 0.0003039
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002669, 0.0002671
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001011, 0.0001012
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003861, 0.0003857
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002702, 0.0002699
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002897, 0.0002894
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001912, 0.0001913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001945, upper bound: 0.0001736
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001904, upper bound: 0.0001809
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001037, 0.0001032
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002630, 0.0002618
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001632, 0.0001624
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003033, 0.0003047
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002675, 0.0002663
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001013, 0.0001009
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003849, 0.0003867
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002693, 0.0002706
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002888, 0.0002901
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001916, 0.0001907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001579, upper bound: 0.0001808
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001559, upper bound: 0.0001804
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000952, 0.0000927
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002416, 0.0002351
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001499, 0.0001459
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002724, 0.0002799
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002458, 0.0002392
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000931, 0.0000906
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003457, 0.0003552
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002419, 0.0002486
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002594, 0.0002665
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001760, 0.0001713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001555, upper bound: 0.0001602
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001512, upper bound: 0.0001657
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000959, 0.0000932
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002433, 0.0002364
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001510, 0.0001467
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002738, 0.0002819
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002475, 0.0002404
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000937, 0.0000911
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003475, 0.0003577
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002432, 0.0002503
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002607, 0.0002684
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001773, 0.0001722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001715, upper bound: 0.0001772
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001690, upper bound: 0.0001805
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000942, 0.0000943
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002389, 0.0002394
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001482, 0.0001485
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002773, 0.0002768
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002430, 0.0002435
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000920, 0.0000922
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003519, 0.0003513
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002463, 0.0002458
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002640, 0.0002635
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001741, 0.0001744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001796, upper bound: 0.0001556
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001579, upper bound: 0.0001808
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000945, 0.0000945
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002398, 0.0002398
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001488, 0.0001488
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002778, 0.0002778
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002439, 0.0002439
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000924, 0.0000924
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003526, 0.0003526
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002467, 0.0002467
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002645, 0.0002645
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001747, 0.0001747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001616, upper bound: 0.0001584
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001559, upper bound: 0.0001622
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001002, 0.0000969
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002543, 0.0002458
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001578, 0.0001525
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002848, 0.0002946
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002587, 0.0002500
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000980, 0.0000947
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003614, 0.0003739
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002529, 0.0002617
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002712, 0.0002806
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001853, 0.0001791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001708, upper bound: 0.0001717
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001676, upper bound: 0.0001768
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001010, 0.0000964
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002563, 0.0002445
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001590, 0.0001517
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002833, 0.0002969
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002607, 0.0002487
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000988, 0.0000942
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003595, 0.0003768
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002516, 0.0002637
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002697, 0.0002827
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001868, 0.0001782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001687, upper bound: 0.0001761
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001641, upper bound: 0.0001814
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001039, 0.0001030
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002636, 0.0002613
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001635, 0.0001621
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003027, 0.0003054
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002681, 0.0002657
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001016, 0.0001007
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003841, 0.0003875
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002688, 0.0002712
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002882, 0.0002907
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001921, 0.0001904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001801, upper bound: 0.0001662
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001729, upper bound: 0.0001678
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001042, 0.0001026
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002643, 0.0002604
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001640, 0.0001616
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003017, 0.0003062
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002689, 0.0002649
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001018, 0.0001003
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003829, 0.0003886
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002679, 0.0002719
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002872, 0.0002915
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001926, 0.0001897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001756, upper bound: 0.0001915
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001720, upper bound: 0.0001959
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001002, 0.0000970
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002543, 0.0002461
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001578, 0.0001527
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002851, 0.0002946
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002587, 0.0002503
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000980, 0.0000948
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003618, 0.0003739
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002532, 0.0002616
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002714, 0.0002805
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001853, 0.0001793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001697, upper bound: 0.0001758
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001711, upper bound: 0.0001757
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001009, 0.0000965
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002561, 0.0002448
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001589, 0.0001519
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002836, 0.0002967
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002605, 0.0002490
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000987, 0.0000943
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003599, 0.0003765
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002519, 0.0002635
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002700, 0.0002825
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001866, 0.0001784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001877, upper bound: 0.0001825
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001699, upper bound: 0.0002002
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001038, 0.0001034
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002635, 0.0002624
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001635, 0.0001628
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003039, 0.0003053
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002680, 0.0002669
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001015, 0.0001011
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003857, 0.0003874
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002699, 0.0002711
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002894, 0.0002907
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001920, 0.0001912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001809, upper bound: 0.0001666
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001729, upper bound: 0.0001680
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001041, 0.0001030
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002641, 0.0002615
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001639, 0.0001622
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003029, 0.0003060
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002687, 0.0002660
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001018, 0.0001007
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003844, 0.0003883
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002690, 0.0002717
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002884, 0.0002914
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001925, 0.0001905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001757, upper bound: 0.0001915
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001717, upper bound: 0.0001958
time: 0.65 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001588, upper bound: 0.0001617
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001769, upper bound: 0.0001731
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001728, upper bound: 0.0001762
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0002002, upper bound: 0.0001699
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001955, upper bound: 0.0001736
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001637, upper bound: 0.0001750
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001602, upper bound: 0.0001738
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001774, upper bound: 0.0001663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001746, upper bound: 0.0001692
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001737, upper bound: 0.0001720
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001707, upper bound: 0.0001761
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001858, upper bound: 0.0001588
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001801, upper bound: 0.0001607
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001631, upper bound: 0.0001751
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001556, upper bound: 0.0001737
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001775, upper bound: 0.0001663
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001734, upper bound: 0.0001685
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001740
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001704, upper bound: 0.0001758
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001803, upper bound: 0.0001559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001808, upper bound: 0.0001579
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001809, upper bound: 0.0001904
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001736, upper bound: 0.0001945
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001860, upper bound: 0.0001577
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001805, upper bound: 0.0001601
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001688, upper bound: 0.0001721
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001663, upper bound: 0.0001754
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001862, upper bound: 0.0001589
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001689, upper bound: 0.0001726
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001801, upper bound: 0.0001607
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001657, upper bound: 0.0001754
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001754, upper bound: 0.0001657
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001726, upper bound: 0.0001689
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001607, upper bound: 0.0001801
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001588, upper bound: 0.0001862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001945, upper bound: 0.0001736
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001904, upper bound: 0.0001809
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001579, upper bound: 0.0001808
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001559, upper bound: 0.0001804
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001555, upper bound: 0.0001602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001512, upper bound: 0.0001657
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001715, upper bound: 0.0001772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001690, upper bound: 0.0001805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001796, upper bound: 0.0001556
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001579, upper bound: 0.0001808
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001616, upper bound: 0.0001584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001559, upper bound: 0.0001622
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001708, upper bound: 0.0001717
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001676, upper bound: 0.0001768
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001687, upper bound: 0.0001761
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001641, upper bound: 0.0001814
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001801, upper bound: 0.0001662
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001729, upper bound: 0.0001678
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001756, upper bound: 0.0001915
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001720, upper bound: 0.0001959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001697, upper bound: 0.0001758
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001711, upper bound: 0.0001757
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001877, upper bound: 0.0001825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001699, upper bound: 0.0002002
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001809, upper bound: 0.0001666
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001729, upper bound: 0.0001680
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001757, upper bound: 0.0001915
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 7, lower bound: -0.0001717, upper bound: 0.0001958

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000965, 0.0001009
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002448, 0.0002561
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001519, 0.0001589
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002967, 0.0002836
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002490, 0.0002605
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000943, 0.0000987
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003765, 0.0003599
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002635, 0.0002519
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002825, 0.0002700
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001784, 0.0001866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001803, upper bound: 0.0001506
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001752, upper bound: 0.0001514
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000970, 0.0001002
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002461, 0.0002543
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001527, 0.0001578
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002946, 0.0002851
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002503, 0.0002587
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000948, 0.0000980
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003739, 0.0003618
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002616, 0.0002532
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002805, 0.0002714
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001793, 0.0001853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001761, upper bound: 0.0001523
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001709, upper bound: 0.0001546
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000986, 0.0000994
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002502, 0.0002523
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001552, 0.0001565
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002922, 0.0002898
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002545, 0.0002566
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000964, 0.0000972
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003709, 0.0003678
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002595, 0.0002574
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002783, 0.0002759
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001823, 0.0001838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001766, upper bound: 0.0001513
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001654
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000818, 0.0000867
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002076, 0.0002201
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001288, 0.0001365
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002550, 0.0002405
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002112, 0.0002239
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000800, 0.0000848
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003236, 0.0003052
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002264, 0.0002136
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002428, 0.0002290
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001513, 0.0001604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001805, upper bound: 0.0001509
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001779, upper bound: 0.0001535
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000835, 0.0000847
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002119, 0.0002150
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001315, 0.0001334
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002490, 0.0002455
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002155, 0.0002186
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000816, 0.0000828
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003160, 0.0003116
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002211, 0.0002180
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002371, 0.0002337
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001544, 0.0001566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001594, upper bound: 0.0001375
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001596, upper bound: 0.0001380
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000979, 0.0000996
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002485, 0.0002528
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001542, 0.0001569
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002929, 0.0002879
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002528, 0.0002572
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000957, 0.0000974
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003717, 0.0003654
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002601, 0.0002557
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002789, 0.0002741
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001811, 0.0001842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001767, upper bound: 0.0001514
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001655
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000942, 0.0000942
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002390, 0.0002391
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001483, 0.0001484
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002770, 0.0002768
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002431, 0.0002432
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000921, 0.0000921
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003516, 0.0003513
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002460, 0.0002459
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002638, 0.0002636
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001741, 0.0001742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001617, upper bound: 0.0001365
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001578, upper bound: 0.0001377
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000940, 0.0000940
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002386, 0.0002386
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001480, 0.0001480
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002764, 0.0002764
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002427, 0.0002427
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000919, 0.0000919
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003508, 0.0003507
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002455, 0.0002454
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002632, 0.0002631
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001738, 0.0001739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001369
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001588, upper bound: 0.0001394
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000982, 0.0000991
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002493, 0.0002516
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001547, 0.0001561
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002915, 0.0002888
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002536, 0.0002559
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000960, 0.0000969
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003699, 0.0003665
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002588, 0.0002565
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002775, 0.0002750
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001816, 0.0001833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001620, upper bound: 0.0001655
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001599, upper bound: 0.0001711
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000989, 0.0000987
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002510, 0.0002504
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001557, 0.0001554
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002901, 0.0002908
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002553, 0.0002547
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000967, 0.0000965
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003682, 0.0003691
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002576, 0.0002583
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002762, 0.0002769
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001829, 0.0001825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001509, upper bound: 0.0001746
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001480, upper bound: 0.0001739
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000811, 0.0000869
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002059, 0.0002206
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001277, 0.0001369
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002556, 0.0002385
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002094, 0.0002244
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000793, 0.0000850
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003244, 0.0003027
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002270, 0.0002118
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002434, 0.0002271
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001500, 0.0001608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001807, upper bound: 0.0001510
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001760, upper bound: 0.0001523
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000828, 0.0000850
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002101, 0.0002157
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001304, 0.0001338
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002498, 0.0002434
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002137, 0.0002194
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000810, 0.0000831
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003171, 0.0003090
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002219, 0.0002162
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002379, 0.0002318
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001531, 0.0001571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001519
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001708, upper bound: 0.0001545
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001018, 0.0001052
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002584, 0.0002670
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001603, 0.0001656
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003093, 0.0002993
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002628, 0.0002716
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000995, 0.0001029
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003925, 0.0003799
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002747, 0.0002658
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002945, 0.0002850
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001883, 0.0001945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001657, upper bound: 0.0001365
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001659, upper bound: 0.0001368
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001018, 0.0001052
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002584, 0.0002670
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001603, 0.0001656
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003093, 0.0002993
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002628, 0.0002716
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000995, 0.0001029
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003925, 0.0003799
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002747, 0.0002658
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002945, 0.0002850
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001883, 0.0001945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Candidate
type: RSZ, layer: 3, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001747, upper bound: 0.0001522
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001709, upper bound: 0.0001552
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000844, 0.0000835
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002142, 0.0002119
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001329, 0.0001315
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002455, 0.0002482
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002179, 0.0002156
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000825, 0.0000817
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003116, 0.0003150
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002180, 0.0002204
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002338, 0.0002363
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001561, 0.0001544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001380, upper bound: 0.0001596
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001375, upper bound: 0.0001594
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000866, 0.0000818
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002198, 0.0002076
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001363, 0.0001288
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002405, 0.0002546
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002235, 0.0002112
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000847, 0.0000800
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003053, 0.0003231
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002136, 0.0002261
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002290, 0.0002424
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001601, 0.0001513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001368, upper bound: 0.0001658
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001365, upper bound: 0.0001657
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000987, 0.0000989
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002504, 0.0002510
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001554, 0.0001557
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002908, 0.0002901
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002547, 0.0002553
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000965, 0.0000967
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003691, 0.0003682
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002583, 0.0002576
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002769, 0.0002762
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001825, 0.0001829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001543
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001683, upper bound: 0.0001551
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000991, 0.0000982
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002516, 0.0002493
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001561, 0.0001547
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002888, 0.0002915
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002559, 0.0002536
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000969, 0.0000960
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003665, 0.0003699
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002565, 0.0002588
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002750, 0.0002775
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001833, 0.0001816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001698, upper bound: 0.0001504
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001710, upper bound: 0.0001580
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000940, 0.0000940
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002386, 0.0002386
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001480, 0.0001480
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002764, 0.0002764
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002427, 0.0002427
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000919, 0.0000919
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003507, 0.0003508
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002454, 0.0002455
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002631, 0.0002632
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001739, 0.0001738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001394, upper bound: 0.0001588
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001369, upper bound: 0.0001621
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000942, 0.0000942
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002391, 0.0002390
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001484, 0.0001483
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002768, 0.0002770
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002432, 0.0002431
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000921, 0.0000921
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003513, 0.0003516
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002458, 0.0002460
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002636, 0.0002638
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001742, 0.0001741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001377, upper bound: 0.0001577
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001366, upper bound: 0.0001616
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001008, 0.0000969
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002558, 0.0002459
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001587, 0.0001525
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002848, 0.0002963
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002602, 0.0002501
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000985, 0.0000947
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003615, 0.0003761
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002529, 0.0002631
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002712, 0.0002821
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001864, 0.0001791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001499, upper bound: 0.0001547
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001463, upper bound: 0.0001606
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001037, 0.0001038
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002630, 0.0002634
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001632, 0.0001634
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003052, 0.0003047
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002676, 0.0002680
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001013, 0.0001015
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003873, 0.0003867
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002710, 0.0002706
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002906, 0.0002901
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001917, 0.0001919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001600, upper bound: 0.0001369
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001517, upper bound: 0.0001374
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001039, 0.0001035
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002637, 0.0002626
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001636, 0.0001629
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0003042, 0.0003055
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002682, 0.0002671
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001016, 0.0001012
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003861, 0.0003877
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002702, 0.0002713
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002896, 0.0002908
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001921, 0.0001913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001394, upper bound: 0.0001588
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001367, upper bound: 0.0001621
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000869, 0.0000811
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002206, 0.0002059
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001369, 0.0001277
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002385, 0.0002556
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002244, 0.0002094
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000850, 0.0000793
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003027, 0.0003244
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002118, 0.0002270
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002271, 0.0002434
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001608, 0.0001500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001634, upper bound: 0.0001631
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001510, upper bound: 0.0001807
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000841, 0.0000843
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002133, 0.0002140
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001323, 0.0001327
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002479, 0.0002471
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002170, 0.0002176
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000822, 0.0000824
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003146, 0.0003136
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002201, 0.0002194
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002360, 0.0002353
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001554, 0.0001559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001597, upper bound: 0.0001409
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001600, upper bound: 0.0001431
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000988, 0.0000984
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002508, 0.0002496
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001556, 0.0001549
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002892, 0.0002906
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002551, 0.0002539
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000966, 0.0000962
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003670, 0.0003688
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002568, 0.0002580
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002754, 0.0002767
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001827, 0.0001819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001524, upper bound: 0.0001723
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001476, upper bound: 0.0001709
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000996, 0.0000979
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002528, 0.0002485
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001569, 0.0001542
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002879, 0.0002929
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002572, 0.0002528
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000974, 0.0000957
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003654, 0.0003717
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002557, 0.0002601
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002741, 0.0002789
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001842, 0.0001811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001530, upper bound: 0.0001737
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001514, upper bound: 0.0001767
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001052, 0.0001015
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002669, 0.0002575
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001656, 0.0001598
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002983, 0.0003092
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002715, 0.0002619
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001028, 0.0000992
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003786, 0.0003924
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002649, 0.0002746
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002841, 0.0002944
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001945, 0.0001876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001668, upper bound: 0.0001555
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001685, upper bound: 0.0001592
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001054, 0.0001012
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002676, 0.0002568
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001660, 0.0001593
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002975, 0.0003100
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002722, 0.0002612
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001031, 0.0000990
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003776, 0.0003934
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002642, 0.0002753
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002833, 0.0002951
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001950, 0.0001871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001514, upper bound: 0.0001752
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0001506, upper bound: 0.0001803
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000842, 0.0000847
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002136, 0.0002150
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001325, 0.0001334
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002491, 0.0002475
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002173, 0.0002187
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000823, 0.0000828
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003161, 0.0003141
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002212, 0.0002198
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002372, 0.0002356
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001557, 0.0001567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001604, upper bound: 0.0001414
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001611, upper bound: 0.0001439
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000989, 0.0000988
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002509, 0.0002507
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001557, 0.0001555
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002904, 0.0002907
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002552, 0.0002550
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000967, 0.0000966
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003686, 0.0003689
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002579, 0.0002582
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002765, 0.0002768
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001828, 0.0001827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001561, upper bound: 0.0001696
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001526, upper bound: 0.0001727
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000996, 0.0000983
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002528, 0.0002494
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001568, 0.0001547
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002890, 0.0002928
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002571, 0.0002537
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000974, 0.0000961
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003667, 0.0003716
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002566, 0.0002600
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002751, 0.0002788
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001842, 0.0001817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001490, upper bound: 0.0001761
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001464, upper bound: 0.0001754
time: 0.64 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001803, upper bound: 0.0001506
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001752, upper bound: 0.0001514
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001761, upper bound: 0.0001523
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001709, upper bound: 0.0001546
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001766, upper bound: 0.0001513
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001654
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001805, upper bound: 0.0001509
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001779, upper bound: 0.0001535
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001594, upper bound: 0.0001375
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001596, upper bound: 0.0001380
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001767, upper bound: 0.0001514
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001617, upper bound: 0.0001365
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001578, upper bound: 0.0001377
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001621, upper bound: 0.0001369
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001588, upper bound: 0.0001394
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001620, upper bound: 0.0001655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001599, upper bound: 0.0001711
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001509, upper bound: 0.0001746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001480, upper bound: 0.0001739
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001807, upper bound: 0.0001510
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001760, upper bound: 0.0001523
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001753, upper bound: 0.0001519
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001708, upper bound: 0.0001545
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001657, upper bound: 0.0001365
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001659, upper bound: 0.0001368
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001747, upper bound: 0.0001522
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001709, upper bound: 0.0001552
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001380, upper bound: 0.0001596
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001375, upper bound: 0.0001594
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001368, upper bound: 0.0001658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001365, upper bound: 0.0001657
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001744, upper bound: 0.0001543
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001683, upper bound: 0.0001551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001698, upper bound: 0.0001504
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001710, upper bound: 0.0001580
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001394, upper bound: 0.0001588
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001369, upper bound: 0.0001621
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001377, upper bound: 0.0001577
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001366, upper bound: 0.0001616
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001499, upper bound: 0.0001547
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001463, upper bound: 0.0001606
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001600, upper bound: 0.0001369
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001517, upper bound: 0.0001374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001394, upper bound: 0.0001588
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001367, upper bound: 0.0001621
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001634, upper bound: 0.0001631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001510, upper bound: 0.0001807
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001597, upper bound: 0.0001409
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001600, upper bound: 0.0001431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001524, upper bound: 0.0001723
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001476, upper bound: 0.0001709
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001530, upper bound: 0.0001737
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001514, upper bound: 0.0001767
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001668, upper bound: 0.0001555
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001685, upper bound: 0.0001592
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001514, upper bound: 0.0001752
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001506, upper bound: 0.0001803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001604, upper bound: 0.0001414
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001611, upper bound: 0.0001439
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001561, upper bound: 0.0001696
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001526, upper bound: 0.0001727
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001490, upper bound: 0.0001761
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.61
Output dim: 7, lower bound: -0.0001464, upper bound: 0.0001754

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000810, 0.0000869
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002056, 0.0002204
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001275, 0.0001368
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002554, 0.0002382
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002091, 0.0002242
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000792, 0.0000849
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003241, 0.0003023
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002268, 0.0002115
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002431, 0.0002268
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001498, 0.0001606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001597, upper bound: 0.0001277
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001598, upper bound: 0.0001288
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000969, 0.0001008
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002459, 0.0002558
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001525, 0.0001587
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002963, 0.0002848
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002501, 0.0002602
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000947, 0.0000985
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003761, 0.0003615
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002631, 0.0002529
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002821, 0.0002712
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001791, 0.0001864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 222

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001598, upper bound: 0.0001287
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001598, upper bound: 0.0001291
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000976, 0.0001002
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002476, 0.0002543
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001536, 0.0001578
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002947, 0.0002868
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002518, 0.0002587
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000954, 0.0000980
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003739, 0.0003640
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002617, 0.0002547
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002806, 0.0002731
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001804, 0.0001853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 222

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001567, upper bound: 0.0001314
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001567, upper bound: 0.0001317
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000964, 0.0001010
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002445, 0.0002563
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001517, 0.0001590
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002969, 0.0002833
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002487, 0.0002607
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000942, 0.0000988
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003768, 0.0003595
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002637, 0.0002516
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002827, 0.0002697
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001782, 0.0001868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 222

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001603, upper bound: 0.0001280
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001606, upper bound: 0.0001293
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0001056, 0.0001011
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002679, 0.0002566
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001662, 0.0001592
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002972, 0.0003104
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002725, 0.0002610
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0001032, 0.0000989
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003772, 0.0003939
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002640, 0.0002756
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002830, 0.0002955
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001952, 0.0001870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 83
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 83

### Candidate
type: RSZ, layer: 3, pos: 222

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001293, upper bound: 0.0001606
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001280, upper bound: 0.0001603
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0017027, -0.0014022, -0.0017027, -0.0014022, -0.0000869, 0.0000810
1: -0.0086315, -0.0078689, -0.0086315, -0.0078689, -0.0002204, 0.0002056
2: 0.0296750, 0.0301481, 0.0296750, 0.0301481, -0.0001368, 0.0001275
3: 0.0032683, 0.0041518, 0.0032683, 0.0041518, -0.0002382, 0.0002554
4: -0.0076727, -0.0068970, -0.0076727, -0.0068970, -0.0002242, 0.0002091
5: 0.0108320, 0.0111258, 0.0108320, 0.0111258, -0.0000849, 0.0000792
6: 0.0045361, 0.0056572, 0.0045361, 0.0056572, -0.0003023, 0.0003241
7: 0.9812334, 0.9820179, 0.9812334, 0.9820179, -0.0002115, 0.0002268
8: -0.0066850, -0.0058438, -0.0066850, -0.0058438, -0.0002268, 0.0002431
9: -0.0011394, -0.0005838, -0.0011394, -0.0005838, -0.0001606, 0.0001498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 194
type: RSZ, layer: 3, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Candidate
type: RSZ, layer: 3, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001288, upper bound: 0.0001598
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0001277, upper bound: 0.0001597
time: 0.67 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001597, upper bound: 0.0001277
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001598, upper bound: 0.0001288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001598, upper bound: 0.0001287
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001598, upper bound: 0.0001291
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001567, upper bound: 0.0001314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001567, upper bound: 0.0001317
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001603, upper bound: 0.0001280
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001606, upper bound: 0.0001293
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001293, upper bound: 0.0001606
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001280, upper bound: 0.0001603
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001288, upper bound: 0.0001598
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.85
Output dim: 7, lower bound: -0.0001277, upper bound: 0.0001597

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.78 + 256.73 = 259.51 seconds
