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
Threshold: 0.00085992


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009998, 0.0009998)
1: (0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001444, 0.0001444)
2: (0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0005527, 0.0005527)
3: (-0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005717, 0.0005717)
4: (0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0006189, 0.0006189)
5: (0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005857, 0.0005857)
6: (-0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0023237, 0.0023237)
7: (0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0031647, 0.0031647)
8: (0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0022293, 0.0022293)
9: (-0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0020236, 0.0020236)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.84 + 1.31 = 3.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0010068, upper bound: 0.0010068

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009300, upper bound: 0.0009509
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0009507, upper bound: 0.0009300
time: 0.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 8, lower bound: -0.0009300, upper bound: 0.0009509
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 8, lower bound: -0.0009507, upper bound: 0.0009300

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009158, 0.0009149
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001323, 0.0001322
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0005058, 0.0005063
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005231, 0.0005237
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005669, 0.0005663
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005359, 0.0005365
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0021264, 0.0021286
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028990, 0.0028960
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020421, 0.0020400
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018518, 0.0018537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008744, upper bound: 0.0008935
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008754, upper bound: 0.0008931
time: 0.54 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009149, 0.0009158
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001322, 0.0001323
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0005063, 0.0005058
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005237, 0.0005231
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005663, 0.0005669
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005365, 0.0005359
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0021286, 0.0021264
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028960, 0.0028990
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020400, 0.0020421
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018537, 0.0018518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008931, upper bound: 0.0008754
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0008744
time: 0.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0008744, upper bound: 0.0008935
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0008754, upper bound: 0.0008931
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0008931, upper bound: 0.0008754
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.0008935, upper bound: 0.0008744

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009006, 0.0009060
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001301, 0.0001309
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0005009, 0.0004979
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005181, 0.0005150
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005575, 0.0005608
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005307, 0.0005276
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0021058, 0.0020932
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028508, 0.0028680
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020081, 0.0020203
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018339, 0.0018229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008664, upper bound: 0.0008836
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008652, upper bound: 0.0008846
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009158, 0.0008996
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001323, 0.0001300
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004974, 0.0005063
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005144, 0.0005237
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005669, 0.0005569
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005270, 0.0005365
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020910, 0.0021286
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028990, 0.0028478
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020421, 0.0020060
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018209, 0.0018537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008666, upper bound: 0.0008829
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008842
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008996, 0.0009044
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001300, 0.0001307
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0005000, 0.0004974
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005172, 0.0005144
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005569, 0.0005598
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005298, 0.0005270
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0021021, 0.0020910
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028478, 0.0028629
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020060, 0.0020167
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018306, 0.0018209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008842, upper bound: 0.0008658
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008829, upper bound: 0.0008667
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009149, 0.0009006
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001322, 0.0001301
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004979, 0.0005058
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005150, 0.0005231
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005663, 0.0005575
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005276, 0.0005359
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020932, 0.0021264
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028960, 0.0028508
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020400, 0.0020081
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018229, 0.0018518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008846, upper bound: 0.0008652
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008836, upper bound: 0.0008664
time: 0.49 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 8, lower bound: -0.0008664, upper bound: 0.0008836
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 8, lower bound: -0.0008652, upper bound: 0.0008846
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 8, lower bound: -0.0008666, upper bound: 0.0008829
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 8, lower bound: -0.0008658, upper bound: 0.0008842
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 8, lower bound: -0.0008842, upper bound: 0.0008658
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 8, lower bound: -0.0008829, upper bound: 0.0008667
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 8, lower bound: -0.0008846, upper bound: 0.0008652
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 8, lower bound: -0.0008836, upper bound: 0.0008664

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008935, 0.0008974
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001291, 0.0001297
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004962, 0.0004940
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005132, 0.0005109
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005531, 0.0005555
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005257, 0.0005234
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020859, 0.0020767
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028283, 0.0028408
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019923, 0.0020011
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018165, 0.0018085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008275, upper bound: 0.0008612
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008428, upper bound: 0.0008374
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008920, 0.0008974
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001289, 0.0001296
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004961, 0.0004932
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005131, 0.0005100
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005522, 0.0005555
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005257, 0.0005225
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020858, 0.0020732
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028236, 0.0028406
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019890, 0.0020010
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018164, 0.0018055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008258, upper bound: 0.0008626
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008415, upper bound: 0.0008399
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009087, 0.0008911
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001313, 0.0001287
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004926, 0.0005024
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005095, 0.0005196
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005625, 0.0005516
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005220, 0.0005323
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020710, 0.0021121
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028765, 0.0028206
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020263, 0.0019869
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018036, 0.0018393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008275, upper bound: 0.0008603
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008431, upper bound: 0.0008374
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009072, 0.0008923
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001311, 0.0001289
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004933, 0.0005016
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005102, 0.0005188
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005616, 0.0005524
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005227, 0.0005315
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020740, 0.0021086
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028718, 0.0028246
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020230, 0.0019897
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018062, 0.0018363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008258, upper bound: 0.0008622
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008422, upper bound: 0.0008399
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008923, 0.0008958
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001289, 0.0001294
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004953, 0.0004933
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005122, 0.0005102
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005524, 0.0005545
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005248, 0.0005227
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020821, 0.0020740
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028246, 0.0028357
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019897, 0.0019975
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018132, 0.0018062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008399, upper bound: 0.0008422
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008622, upper bound: 0.0008258
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008911, 0.0008960
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001287, 0.0001294
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004954, 0.0004926
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005124, 0.0005095
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005516, 0.0005547
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005249, 0.0005220
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020826, 0.0020710
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028206, 0.0028363
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019869, 0.0019980
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018136, 0.0018036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008374, upper bound: 0.0008431
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008603, upper bound: 0.0008275
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009076, 0.0008920
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001311, 0.0001289
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004932, 0.0005018
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005100, 0.0005190
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005618, 0.0005522
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005225, 0.0005317
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020732, 0.0021094
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028729, 0.0028236
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020237, 0.0019890
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018055, 0.0018370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008399, upper bound: 0.0008415
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008626, upper bound: 0.0008258
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009063, 0.0008935
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001309, 0.0001291
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004940, 0.0005011
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005109, 0.0005182
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005610, 0.0005531
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005234, 0.0005309
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020767, 0.0021065
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028688, 0.0028283
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020209, 0.0019923
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018085, 0.0018344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0008374, upper bound: 0.0008428
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008612, upper bound: 0.0008275
time: 0.49 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008275, upper bound: 0.0008612
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008428, upper bound: 0.0008374
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008258, upper bound: 0.0008626
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008415, upper bound: 0.0008399
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008275, upper bound: 0.0008603
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008431, upper bound: 0.0008374
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008258, upper bound: 0.0008622
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008422, upper bound: 0.0008399
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008399, upper bound: 0.0008422
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008622, upper bound: 0.0008258
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008374, upper bound: 0.0008431
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008603, upper bound: 0.0008275
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008399, upper bound: 0.0008415
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008626, upper bound: 0.0008258
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008374, upper bound: 0.0008428
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.82
Output dim: 8, lower bound: -0.0008612, upper bound: 0.0008275

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008811, 0.0008922
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001273, 0.0001289
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004933, 0.0004872
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005102, 0.0005038
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005454, 0.0005523
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005226, 0.0005162
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020737, 0.0020480
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0027892, 0.0028242
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019648, 0.0019894
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018059, 0.0017835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007114, upper bound: 0.0007663
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007378, upper bound: 0.0007495
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008797, 0.0008921
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001271, 0.0001289
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004932, 0.0004863
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005101, 0.0005030
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005445, 0.0005523
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005226, 0.0005153
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020736, 0.0020446
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0027845, 0.0028241
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019615, 0.0019893
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0018058, 0.0017805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007094, upper bound: 0.0007671
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007378, upper bound: 0.0007511
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008945, 0.0008858
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001292, 0.0001280
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004898, 0.0004946
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005065, 0.0005115
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005537, 0.0005484
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005189, 0.0005240
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020590, 0.0020792
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028317, 0.0028041
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019947, 0.0019753
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0017930, 0.0018106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007114, upper bound: 0.0007638
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007378, upper bound: 0.0007488
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008931, 0.0008870
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001290, 0.0001281
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004904, 0.0004938
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005072, 0.0005107
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005528, 0.0005490
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005196, 0.0005232
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020615, 0.0020757
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028270, 0.0028076
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019914, 0.0019778
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0017953, 0.0018076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007094, upper bound: 0.0007658
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007378, upper bound: 0.0007504
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008870, 0.0008835
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001281, 0.0001276
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004885, 0.0004904
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005052, 0.0005072
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005490, 0.0005469
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005175, 0.0005196
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020535, 0.0020615
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028076, 0.0027967
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019778, 0.0019700
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0017883, 0.0017953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007504, upper bound: 0.0007378
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007658, upper bound: 0.0007094
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008858, 0.0008837
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001280, 0.0001277
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004886, 0.0004898
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005053, 0.0005065
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005484, 0.0005470
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005177, 0.0005189
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020539, 0.0020590
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028041, 0.0027973
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0019753, 0.0019705
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0017887, 0.0017930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007488, upper bound: 0.0007378
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007638, upper bound: 0.0007114
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0009004, 0.0008797
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001301, 0.0001271
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004863, 0.0004978
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005030, 0.0005148
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005573, 0.0005445
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005153, 0.0005274
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020446, 0.0020927
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028501, 0.0027845
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020076, 0.0019615
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0017805, 0.0018224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007511, upper bound: 0.0007378
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007671, upper bound: 0.0007094
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068229, 0.0083016, 0.0068229, 0.0083016, -0.0008993, 0.0008811
1: 0.0023080, 0.0025216, 0.0023080, 0.0025216, -0.0001299, 0.0001273
2: 0.0097701, 0.0105877, 0.0097701, 0.0105877, -0.0004872, 0.0004972
3: -0.0045757, -0.0037302, -0.0045757, -0.0037302, -0.0005038, 0.0005142
4: 0.0000012, 0.0009165, 0.0000012, 0.0009165, -0.0005567, 0.0005454
5: 0.0032460, 0.0041122, 0.0032460, 0.0041122, -0.0005162, 0.0005268
6: -0.0094212, -0.0059843, -0.0094212, -0.0059843, -0.0020480, 0.0020901
7: 0.0055933, 0.0102742, 0.0055933, 0.0102742, -0.0028465, 0.0027892
8: 0.9931539, 0.9964512, 0.9931539, 0.9964512, -0.0020052, 0.0019648
9: -0.0126659, -0.0096729, -0.0126659, -0.0096729, -0.0017835, 0.0018202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007495, upper bound: 0.0007378
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007663, upper bound: 0.0007114
time: 0.50 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007114, upper bound: 0.0007663
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007378, upper bound: 0.0007495
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007094, upper bound: 0.0007671
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007378, upper bound: 0.0007511
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007114, upper bound: 0.0007638
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007378, upper bound: 0.0007488
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007094, upper bound: 0.0007658
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007378, upper bound: 0.0007504
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007504, upper bound: 0.0007378
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007658, upper bound: 0.0007094
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007488, upper bound: 0.0007378
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007638, upper bound: 0.0007114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007511, upper bound: 0.0007378
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007671, upper bound: 0.0007094
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007495, upper bound: 0.0007378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 8, lower bound: -0.0007663, upper bound: 0.0007114

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.15 + 64.10 = 67.26 seconds
