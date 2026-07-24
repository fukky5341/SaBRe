## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0012732


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0018400, 0.0018400)
1: (0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002658, 0.0002658)
2: (0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0010173, 0.0010173)
3: (-0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0010521, 0.0010521)
4: (-0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0011390, 0.0011390)
5: (0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010778, 0.0010778)
6: (-0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0042766, 0.0042766)
7: (-0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0058243, 0.0058243)
8: (0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0041028, 0.0041028)
9: (-0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0037242, 0.0037242)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 1.71 = 3.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0022648, upper bound: 0.0022648

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020402, upper bound: 0.0020402
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020402, upper bound: 0.0020402
time: 0.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 8, lower bound: -0.0020402, upper bound: 0.0020402
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 8, lower bound: -0.0020402, upper bound: 0.0020402

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0018292, 0.0018204
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002643, 0.0002630
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0010065, 0.0010113
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0010409, 0.0010460
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0011323, 0.0011269
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010664, 0.0010716
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0042312, 0.0042517
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0057904, 0.0057625
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0040789, 0.0040592
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0036847, 0.0037025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020208, upper bound: 0.0020268
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020268, upper bound: 0.0020208
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0018400, 0.0018292
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002658, 0.0002643
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0010113, 0.0010173
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0010460, 0.0010521
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0011390, 0.0011323
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010716, 0.0010778
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0042517, 0.0042766
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0058243, 0.0057904
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0041028, 0.0040789
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0037025, 0.0037242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020208, upper bound: 0.0020268
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020268, upper bound: 0.0020208
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 8, lower bound: -0.0020208, upper bound: 0.0020268
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 8, lower bound: -0.0020268, upper bound: 0.0020208
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 8, lower bound: -0.0020208, upper bound: 0.0020268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 8, lower bound: -0.0020268, upper bound: 0.0020208

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0018313, 0.0018280
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002646, 0.0002641
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0010107, 0.0010125
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0010453, 0.0010472
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0011336, 0.0011316
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010708, 0.0010728
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0042488, 0.0042565
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0057970, 0.0057865
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0040836, 0.0040761
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0037000, 0.0037068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018916, upper bound: 0.0018991
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018945, upper bound: 0.0018965
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0018372, 0.0018225
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002654, 0.0002633
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0010076, 0.0010157
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0010421, 0.0010505
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0011372, 0.0011282
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010676, 0.0010762
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0042360, 0.0042701
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0058155, 0.0057691
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0040966, 0.0040639
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0036889, 0.0037186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018965, upper bound: 0.0018945
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018991, upper bound: 0.0018916
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0018419, 0.0018372
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002661, 0.0002654
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0010157, 0.0010184
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0010505, 0.0010532
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0011402, 0.0011372
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010762, 0.0010790
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0042701, 0.0042811
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0058305, 0.0058155
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0041072, 0.0040966
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0037186, 0.0037282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018916, upper bound: 0.0018991
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018945, upper bound: 0.0018965
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0018478, 0.0018313
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002669, 0.0002646
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0010125, 0.0010216
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0010472, 0.0010566
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0011438, 0.0011336
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010728, 0.0010824
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0042565, 0.0042947
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0058490, 0.0057970
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0041202, 0.0040836
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0037068, 0.0037400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018965, upper bound: 0.0018945
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018991, upper bound: 0.0018916
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0018916, upper bound: 0.0018991
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0018945, upper bound: 0.0018965
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0018965, upper bound: 0.0018945
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0018991, upper bound: 0.0018916
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0018916, upper bound: 0.0018991
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0018945, upper bound: 0.0018965
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0018965, upper bound: 0.0018945
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 8, lower bound: -0.0018991, upper bound: 0.0018916

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017283, 0.0017195
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002497, 0.0002484
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009506, 0.0009555
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009832, 0.0009883
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010699, 0.0010644
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010073, 0.0010124
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039965, 0.0040171
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054709, 0.0054429
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038538, 0.0038341
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034803, 0.0034983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017203, 0.0017250
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002485, 0.0002492
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009537, 0.0009511
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009864, 0.0009837
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010649, 0.0010678
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010105, 0.0010077
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0040093, 0.0039984
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054455, 0.0054603
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038359, 0.0038464
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034915, 0.0034820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017342, 0.0017148
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002505, 0.0002477
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009481, 0.0009588
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009806, 0.0009916
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010735, 0.0010615
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010045, 0.0010159
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039857, 0.0040307
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054894, 0.0054282
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038668, 0.0038237
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034709, 0.0035101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017262, 0.0017195
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002494, 0.0002484
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009507, 0.0009544
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009832, 0.0009870
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010685, 0.0010644
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010073, 0.0010112
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039966, 0.0040121
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054642, 0.0054430
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038491, 0.0038341
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034804, 0.0034939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017382, 0.0017262
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002511, 0.0002494
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009544, 0.0009610
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009870, 0.0009939
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010760, 0.0010685
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010112, 0.0010183
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0040121, 0.0040402
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0055023, 0.0054642
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038760, 0.0038491
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034939, 0.0035183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017302, 0.0017342
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002500, 0.0002505
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009588, 0.0009566
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009916, 0.0009893
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010710, 0.0010735
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010159, 0.0010136
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0040306, 0.0040215
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054769, 0.0054894
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038581, 0.0038668
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0035101, 0.0035021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017441, 0.0017203
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002520, 0.0002485
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009511, 0.0009643
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009837, 0.0009973
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010796, 0.0010649
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010077, 0.0010217
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039984, 0.0040537
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0055208, 0.0054455
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038890, 0.0038359
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034820, 0.0035302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017361, 0.0017283
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002508, 0.0002497
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009555, 0.0009599
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009883, 0.0009927
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010747, 0.0010699
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010124, 0.0010170
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0040171, 0.0040352
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054956, 0.0054709
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038712, 0.0038538
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034983, 0.0035140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013867, upper bound: 0.0013869
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.70
Output dim: 8, lower bound: -0.0013869, upper bound: 0.0013867

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017101, 0.0017092
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002471, 0.0002469
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009450, 0.0009455
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009773, 0.0009779
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010586, 0.0010580
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010013, 0.0010018
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039727, 0.0039748
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054133, 0.0054105
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038132, 0.0038113
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034596, 0.0034614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012917, upper bound: 0.0013076
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017164, 0.0017013
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002480, 0.0002458
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009406, 0.0009490
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009728, 0.0009815
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010625, 0.0010531
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009966, 0.0010055
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039542, 0.0039894
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054332, 0.0053853
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038273, 0.0037935
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034435, 0.0034741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012917, upper bound: 0.0013076
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017021, 0.0017138
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002459, 0.0002476
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009475, 0.0009410
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009799, 0.0009733
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010536, 0.0010608
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010039, 0.0009971
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039832, 0.0039561
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0053879, 0.0054248
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0037953, 0.0038214
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034688, 0.0034451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012918, upper bound: 0.0013076
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017085, 0.0017068
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002468, 0.0002466
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009436, 0.0009446
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009759, 0.0009769
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010576, 0.0010565
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009998, 0.0010008
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039670, 0.0039709
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054080, 0.0054027
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038095, 0.0038058
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034546, 0.0034580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012918, upper bound: 0.0013076
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017159, 0.0017045
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002479, 0.0002463
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009424, 0.0009487
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009747, 0.0009812
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010622, 0.0010551
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009985, 0.0010052
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039618, 0.0039883
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054318, 0.0053956
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038262, 0.0038008
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034501, 0.0034732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012918
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017222, 0.0016966
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002488, 0.0002451
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009380, 0.0009522
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009701, 0.0009848
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010661, 0.0010502
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009939, 0.0010089
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039434, 0.0040029
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054517, 0.0053706
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038403, 0.0037831
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034341, 0.0034859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012918
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017080, 0.0017086
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002468, 0.0002468
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009446, 0.0009443
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009770, 0.0009766
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010573, 0.0010576
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010009, 0.0010005
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039712, 0.0039698
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054065, 0.0054085
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038085, 0.0038098
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034583, 0.0034571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012917
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017144, 0.0017013
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002477, 0.0002458
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009406, 0.0009478
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009728, 0.0009803
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010612, 0.0010531
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009966, 0.0010043
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039542, 0.0039847
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054268, 0.0053853
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038227, 0.0037935
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034435, 0.0034700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012917
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017200, 0.0017144
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002485, 0.0002477
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009478, 0.0009509
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009803, 0.0009835
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010647, 0.0010612
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010043, 0.0010075
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039847, 0.0039977
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054445, 0.0054268
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038352, 0.0038227
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034700, 0.0034813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012917, upper bound: 0.0013076
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017263, 0.0017080
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002494, 0.0002468
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009443, 0.0009544
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009766, 0.0009871
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010686, 0.0010573
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010005, 0.0010112
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039698, 0.0040123
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054644, 0.0054065
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038492, 0.0038085
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034571, 0.0034941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012917, upper bound: 0.0013076
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017119, 0.0017222
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002473, 0.0002488
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009522, 0.0009465
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009848, 0.0009789
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010597, 0.0010661
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010089, 0.0010028
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0040029, 0.0039790
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054190, 0.0054517
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038173, 0.0038403
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034859, 0.0034651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012918, upper bound: 0.0013076
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017183, 0.0017159
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002482, 0.0002479
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009487, 0.0009500
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009812, 0.0009825
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010637, 0.0010622
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010052, 0.0010066
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039883, 0.0039938
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054392, 0.0054318
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038315, 0.0038262
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034732, 0.0034780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012918, upper bound: 0.0013076
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017258, 0.0017085
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002493, 0.0002468
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009446, 0.0009541
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009769, 0.0009868
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010683, 0.0010576
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010008, 0.0010110
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039709, 0.0040112
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054629, 0.0054080
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038482, 0.0038095
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034580, 0.0034932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012918
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017321, 0.0017021
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002502, 0.0002459
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009410, 0.0009576
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009733, 0.0009904
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010722, 0.0010536
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009971, 0.0010147
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039561, 0.0040258
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054828, 0.0053879
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038622, 0.0037953
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034451, 0.0035059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012918
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017178, 0.0017164
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002482, 0.0002480
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009490, 0.0009497
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009815, 0.0009823
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010634, 0.0010625
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010055, 0.0010063
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039894, 0.0039927
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054377, 0.0054332
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038304, 0.0038273
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034741, 0.0034770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012917
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0017242, 0.0017101
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002491, 0.0002471
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0009455, 0.0009533
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009779, 0.0009859
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010673, 0.0010586
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0010018, 0.0010100
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0039748, 0.0040075
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0054579, 0.0054133
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0038447, 0.0038132
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0034614, 0.0034900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012917
time: 0.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 8.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012917, upper bound: 0.0013076
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012917, upper bound: 0.0013076
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012918, upper bound: 0.0013076
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012918, upper bound: 0.0013076
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012918
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012918
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012917
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012917
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012917, upper bound: 0.0013076
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012917, upper bound: 0.0013076
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012918, upper bound: 0.0013076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012918, upper bound: 0.0013076
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013072, upper bound: 0.0012916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012918
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012918
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012917
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0012916, upper bound: 0.0013072
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.49
Output dim: 8, lower bound: -0.0013076, upper bound: 0.0012917

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015908, 0.0016012
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002298, 0.0002313
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008853, 0.0008795
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009156, 0.0009096
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009847, 0.0009912
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009380, 0.0009319
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037217, 0.0036975
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050357, 0.0050686
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035473, 0.0035704
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032410, 0.0032200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016014, 0.0015899
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002314, 0.0002297
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008790, 0.0008854
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009091, 0.0009157
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009913, 0.0009842
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009314, 0.0009381
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036955, 0.0037221
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050692, 0.0050329
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035709, 0.0035453
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032182, 0.0032414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015971, 0.0015927
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002307, 0.0002301
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008805, 0.0008830
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009107, 0.0009132
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009886, 0.0009859
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009330, 0.0009356
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037018, 0.0037122
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050556, 0.0050415
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035613, 0.0035513
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032237, 0.0032327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016082, 0.0015820
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002323, 0.0002286
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008746, 0.0008891
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009046, 0.0009196
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009955, 0.0009793
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009267, 0.0009421
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036770, 0.0037380
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050908, 0.0050077
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035861, 0.0035275
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032021, 0.0032552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015828, 0.0016048
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002287, 0.0002318
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008873, 0.0008751
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009176, 0.0009051
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009798, 0.0009934
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009401, 0.0009272
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037300, 0.0036789
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050103, 0.0050799
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035293, 0.0035784
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032482, 0.0032037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015929, 0.0015945
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002301, 0.0002304
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008815, 0.0008807
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009117, 0.0009109
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009861, 0.0009870
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009340, 0.0009331
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037060, 0.0037024
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050424, 0.0050473
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035520, 0.0035554
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032274, 0.0032242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015892, 0.0015971
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002296, 0.0002307
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008830, 0.0008786
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009132, 0.0009087
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009837, 0.0009886
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009356, 0.0009309
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037120, 0.0036937
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050305, 0.0050555
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035436, 0.0035612
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032326, 0.0032166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016006, 0.0015875
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002312, 0.0002293
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008777, 0.0008850
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009077, 0.0009153
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009908, 0.0009827
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009299, 0.0009377
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036898, 0.0037203
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050668, 0.0050251
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035691, 0.0035398
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032132, 0.0032398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015967, 0.0015955
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002307, 0.0002305
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008821, 0.0008828
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009123, 0.0009130
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009884, 0.0009876
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009346, 0.0009353
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037083, 0.0037111
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050542, 0.0050504
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035603, 0.0035576
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032294, 0.0032318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016076, 0.0015852
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002323, 0.0002290
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008764, 0.0008888
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009065, 0.0009193
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009952, 0.0009813
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009286, 0.0009418
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036845, 0.0037366
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050889, 0.0050180
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035847, 0.0035348
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032086, 0.0032540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016030, 0.0015869
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002316, 0.0002293
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008773, 0.0008862
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009074, 0.0009166
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009923, 0.0009823
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009296, 0.0009390
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036883, 0.0037257
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050741, 0.0050232
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035743, 0.0035384
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032120, 0.0032445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016143, 0.0015773
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002332, 0.0002279
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008721, 0.0008925
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009019, 0.0009231
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009993, 0.0009764
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009240, 0.0009456
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036662, 0.0037520
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0051099, 0.0049930
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035995, 0.0035172
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0031927, 0.0032674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015887, 0.0015992
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002295, 0.0002310
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008842, 0.0008784
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009144, 0.0009084
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009834, 0.0009899
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009368, 0.0009307
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037170, 0.0036926
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050290, 0.0050622
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035425, 0.0035659
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032369, 0.0032157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015994, 0.0015893
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002311, 0.0002296
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008787, 0.0008843
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009088, 0.0009146
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009901, 0.0009838
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009310, 0.0009369
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036940, 0.0037175
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050630, 0.0050309
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035665, 0.0035439
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032169, 0.0032374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015951, 0.0015913
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002304, 0.0002299
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008798, 0.0008819
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009099, 0.0009121
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009874, 0.0009850
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009322, 0.0009344
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036985, 0.0037074
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050492, 0.0050371
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035568, 0.0035482
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032208, 0.0032286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016070, 0.0015820
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002322, 0.0002286
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008746, 0.0008885
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009046, 0.0009189
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009947, 0.0009793
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009267, 0.0009414
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036770, 0.0037351
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050868, 0.0050078
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035833, 0.0035276
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032021, 0.0032527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016020, 0.0016070
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002314, 0.0002322
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008885, 0.0008857
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009189, 0.0009160
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009917, 0.0009947
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009414, 0.0009384
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037351, 0.0037234
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050710, 0.0050868
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035721, 0.0035833
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032527, 0.0032425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016126, 0.0015951
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002330, 0.0002304
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008819, 0.0008915
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009121, 0.0009221
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009982, 0.0009874
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009344, 0.0009446
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037074, 0.0037480
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0051045, 0.0050492
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035957, 0.0035568
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032286, 0.0032640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016083, 0.0015994
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002323, 0.0002311
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008843, 0.0008892
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009146, 0.0009196
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009956, 0.0009901
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009369, 0.0009421
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037175, 0.0037381
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050909, 0.0050630
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035862, 0.0035665
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032374, 0.0032553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016194, 0.0015887
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002340, 0.0002295
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008784, 0.0008953
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009084, 0.0009260
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010024, 0.0009834
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009307, 0.0009486
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036926, 0.0037639
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0051261, 0.0050290
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0036109, 0.0035425
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032157, 0.0032778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015939, 0.0016143
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002303, 0.0002332
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008925, 0.0008813
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009231, 0.0009114
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009867, 0.0009993
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009456, 0.0009337
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037520, 0.0037048
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050456, 0.0051099
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035542, 0.0035995
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032674, 0.0032263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016041, 0.0016030
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002317, 0.0002316
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008862, 0.0008869
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009166, 0.0009172
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009930, 0.0009923
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009390, 0.0009397
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037257, 0.0037284
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050777, 0.0050741
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035768, 0.0035743
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032445, 0.0032468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016003, 0.0016076
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002312, 0.0002323
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008888, 0.0008848
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009193, 0.0009151
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009906, 0.0009952
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009418, 0.0009375
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037366, 0.0037196
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050658, 0.0050889
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035684, 0.0035847
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032540, 0.0032392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016118, 0.0015967
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002329, 0.0002307
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008828, 0.0008911
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009130, 0.0009216
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009977, 0.0009884
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009353, 0.0009442
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037111, 0.0037463
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0051021, 0.0050542
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035940, 0.0035603
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032318, 0.0032624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016078, 0.0016006
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002323, 0.0002312
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008850, 0.0008889
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009153, 0.0009194
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009953, 0.0009908
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009377, 0.0009419
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037203, 0.0037370
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050895, 0.0050668
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035851, 0.0035691
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032398, 0.0032544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016188, 0.0015892
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002339, 0.0002296
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008786, 0.0008950
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009087, 0.0009256
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010021, 0.0009837
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009309, 0.0009483
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036937, 0.0037625
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0051242, 0.0050305
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0036096, 0.0035436
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032166, 0.0032766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016141, 0.0015929
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002332, 0.0002301
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008807, 0.0008924
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009109, 0.0009230
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009992, 0.0009861
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009331, 0.0009455
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037024, 0.0037516
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0051094, 0.0050424
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035992, 0.0035520
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032242, 0.0032671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016254, 0.0015828
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002348, 0.0002287
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008751, 0.0008987
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009051, 0.0009294
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010062, 0.0009798
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009272, 0.0009522
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036789, 0.0037779
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0051452, 0.0050103
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0036244, 0.0035293
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032037, 0.0032900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0015999, 0.0016082
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002311, 0.0002323
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008891, 0.0008845
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009196, 0.0009148
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009903, 0.0009955
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009421, 0.0009372
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037380, 0.0037185
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050643, 0.0050908
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035674, 0.0035861
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032552, 0.0032382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016106, 0.0015971
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002327, 0.0002307
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008830, 0.0008905
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009132, 0.0009210
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009970, 0.0009886
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009356, 0.0009435
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037122, 0.0037435
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050983, 0.0050556
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035913, 0.0035613
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032327, 0.0032600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016062, 0.0016014
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002321, 0.0002314
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008854, 0.0008880
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009157, 0.0009185
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0009943, 0.0009913
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009381, 0.0009409
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0037221, 0.0037333
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0050845, 0.0050692
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0035816, 0.0035709
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032414, 0.0032512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0022527, 0.0050166, 0.0022527, 0.0050166, -0.0016181, 0.0015908
1: 0.0016478, 0.0020471, 0.0016478, 0.0020471, -0.0002338, 0.0002298
2: 0.0115863, 0.0131144, 0.0115863, 0.0131144, -0.0008795, 0.0008946
3: -0.0026974, -0.0011170, -0.0026974, -0.0011170, -0.0009096, 0.0009253
4: -0.0028278, -0.0011169, -0.0028278, -0.0011169, -0.0010017, 0.0009847
5: 0.0051703, 0.0067894, 0.0051703, 0.0067894, -0.0009319, 0.0009479
6: -0.0017860, 0.0046380, -0.0017860, 0.0046380, -0.0036975, 0.0037610
7: -0.0088733, -0.0001244, -0.0088733, -0.0001244, -0.0051221, 0.0050357
8: 0.9829633, 0.9891263, 0.9829633, 0.9891263, -0.0036081, 0.0035473
9: -0.0060168, -0.0004225, -0.0060168, -0.0004225, -0.0032200, 0.0032752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
time: 0.71 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 8.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012103
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012097, upper bound: 0.0012071
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0011942, upper bound: 0.0012238
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012071, upper bound: 0.0012097
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012103, upper bound: 0.0012071
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 8.68
Output dim: 8, lower bound: -0.0012238, upper bound: 0.0011942

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.12 + 452.73 = 455.85 seconds
