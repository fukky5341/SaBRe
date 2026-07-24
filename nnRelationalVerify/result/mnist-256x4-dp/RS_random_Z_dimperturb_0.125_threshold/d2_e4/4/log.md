## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00058656


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0006833, 0.0006833)
1: (0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000987, 0.0000987)
2: (0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003778, 0.0003778)
3: (-0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003907, 0.0003907)
4: (-0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0004230, 0.0004230)
5: (0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0004003, 0.0004003)
6: (-0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0015882, 0.0015882)
7: (-0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0021630, 0.0021630)
8: (0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0015237, 0.0015237)
9: (-0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0013831, 0.0013831)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.31 = 2.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0006818, upper bound: 0.0006819

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006622, upper bound: 0.0006679
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006679, upper bound: 0.0006622
time: 0.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.95 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.95
Output dim: 8, lower bound: -0.0006622, upper bound: 0.0006679
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.95
Output dim: 8, lower bound: -0.0006679, upper bound: 0.0006622

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0006771, 0.0006790
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000978, 0.0000981
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003754, 0.0003744
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003883, 0.0003872
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0004191, 0.0004203
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003978, 0.0003966
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0015782, 0.0015738
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0021434, 0.0021494
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0015098, 0.0015141
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0013744, 0.0013705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006046, upper bound: 0.0006280
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006215, upper bound: 0.0006119
time: 0.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0006790, 0.0006771
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000981, 0.0000978
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003744, 0.0003754
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003872, 0.0003883
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0004203, 0.0004191
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003966, 0.0003978
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0015738, 0.0015782
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0021494, 0.0021434
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0015141, 0.0015098
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0013705, 0.0013744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0006288
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006324, upper bound: 0.0005483
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 8, lower bound: -0.0006046, upper bound: 0.0006280
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 8, lower bound: -0.0006215, upper bound: 0.0006119
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0006288
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 8, lower bound: -0.0006324, upper bound: 0.0005483

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005580, 0.0005653
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000806, 0.0000817
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003126, 0.0003085
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003233, 0.0003191
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003454, 0.0003500
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003312, 0.0003269
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0013140, 0.0012969
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0017663, 0.0017896
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0012442, 0.0012606
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011443, 0.0011294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004933, upper bound: 0.0005928
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005704, upper bound: 0.0005045
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005648, 0.0005599
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000816, 0.0000809
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003095, 0.0003122
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003201, 0.0003229
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003496, 0.0003466
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003280, 0.0003308
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0013013, 0.0013127
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0017877, 0.0017723
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0012593, 0.0012484
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011332, 0.0011431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005993, upper bound: 0.0005893
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005997, upper bound: 0.0005883
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005760, 0.0006039
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000832, 0.0000872
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003339, 0.0003184
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003453, 0.0003293
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003565, 0.0003738
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003537, 0.0003374
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0014035, 0.0013387
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0018232, 0.0019115
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0012843, 0.0013465
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0012223, 0.0011658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005218, upper bound: 0.0006071
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005274, upper bound: 0.0006005
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0006062, 0.0005741
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000876, 0.0000829
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003174, 0.0003351
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003283, 0.0003466
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003752, 0.0003554
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003363, 0.0003551
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0013343, 0.0014089
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0019188, 0.0018172
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0013516, 0.0012801
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011620, 0.0012269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006019, upper bound: 0.0005273
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006127, upper bound: 0.0005218
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0004933, upper bound: 0.0005928
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0005704, upper bound: 0.0005045
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0005993, upper bound: 0.0005893
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0005997, upper bound: 0.0005883
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0005218, upper bound: 0.0006071
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0005274, upper bound: 0.0006005
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0006019, upper bound: 0.0005273
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0006127, upper bound: 0.0005218

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0004555, 0.0004906
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000658, 0.0000709
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0002712, 0.0002518
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0002805, 0.0002605
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0002820, 0.0003037
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0002874, 0.0002668
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0011403, 0.0010588
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0014419, 0.0015530
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0010157, 0.0010940
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0009930, 0.0009220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004238, upper bound: 0.0005735
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004724, upper bound: 0.0005071
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005576, 0.0005535
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000806, 0.0000800
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003060, 0.0003083
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003165, 0.0003188
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003451, 0.0003426
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003243, 0.0003266
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0012866, 0.0012959
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0017649, 0.0017522
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0012433, 0.0012343
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011204, 0.0011286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005751, upper bound: 0.0005657
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005776, upper bound: 0.0005603
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005593, 0.0005527
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000808, 0.0000798
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003056, 0.0003092
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003160, 0.0003198
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003462, 0.0003421
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003238, 0.0003277
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0012846, 0.0013000
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0017705, 0.0017495
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0012472, 0.0012324
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011187, 0.0011321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004828, upper bound: 0.0005539
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005644, upper bound: 0.0004689
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005600, 0.0005893
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000809, 0.0000851
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003258, 0.0003096
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003370, 0.0003202
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003466, 0.0003648
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003452, 0.0003280
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0013697, 0.0013015
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0017725, 0.0018653
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0012486, 0.0013140
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011928, 0.0011334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004667, upper bound: 0.0005656
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004785, upper bound: 0.0005491
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005607, 0.0005878
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000810, 0.0000849
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003250, 0.0003100
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003361, 0.0003206
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003471, 0.0003639
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003444, 0.0003284
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0013663, 0.0013031
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0017748, 0.0018608
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0012502, 0.0013108
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011898, 0.0011348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004544, upper bound: 0.0005807
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005058, upper bound: 0.0005231
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005901, 0.0005592
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000853, 0.0000808
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003092, 0.0003263
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003198, 0.0003375
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003653, 0.0003462
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003276, 0.0003457
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0012997, 0.0013717
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0018681, 0.0017701
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0013159, 0.0012469
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011319, 0.0011945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005782, upper bound: 0.0005044
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005782, upper bound: 0.0005019
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0038266, 0.0047220, 0.0038266, 0.0047220, -0.0005906, 0.0005581
1: 0.0018751, 0.0020045, 0.0018751, 0.0020045, -0.0000853, 0.0000806
2: 0.0117492, 0.0122442, 0.0117492, 0.0122442, -0.0003085, 0.0003265
3: -0.0025289, -0.0020169, -0.0025289, -0.0020169, -0.0003191, 0.0003377
4: -0.0018536, -0.0012993, -0.0018536, -0.0012993, -0.0003656, 0.0003454
5: 0.0053429, 0.0058674, 0.0053429, 0.0058674, -0.0003269, 0.0003460
6: -0.0011013, 0.0009799, -0.0011013, 0.0009799, -0.0012971, 0.0013727
7: -0.0038913, -0.0010568, -0.0038913, -0.0010568, -0.0018694, 0.0017665
8: 0.9864727, 0.9884694, 0.9864727, 0.9884694, -0.0013169, 0.0012444
9: -0.0054206, -0.0036081, -0.0054206, -0.0036081, -0.0011295, 0.0011954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005583, upper bound: 0.0004785
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005732, upper bound: 0.0004667
time: 0.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0004238, upper bound: 0.0005735
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0004724, upper bound: 0.0005071
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0005751, upper bound: 0.0005657
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0005776, upper bound: 0.0005603
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0004828, upper bound: 0.0005539
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0005644, upper bound: 0.0004689
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0004667, upper bound: 0.0005656
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0004785, upper bound: 0.0005491
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0004544, upper bound: 0.0005807
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0005058, upper bound: 0.0005231
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0005782, upper bound: 0.0005044
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0005782, upper bound: 0.0005019
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0005583, upper bound: 0.0004785
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 8, lower bound: -0.0005732, upper bound: 0.0004667

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.90 + 32.39 = 35.29 seconds
