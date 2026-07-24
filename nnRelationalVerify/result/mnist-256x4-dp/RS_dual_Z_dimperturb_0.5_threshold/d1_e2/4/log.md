## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00037578


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698)
1: (-0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682)
2: (0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219)
3: (-0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0052424, 0.0052424)
4: (0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172)
5: (0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716)
6: (0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173)
7: (-0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560)
8: (0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981)
9: (0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023394, 0.0023394)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.12 + 1.75 = 2.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0006282, upper bound: 0.0006282

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006275, upper bound: 0.0006276
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006276, upper bound: 0.0006276
time: 0.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.02
Output dim: 2, lower bound: -0.0006275, upper bound: 0.0006276
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.02
Output dim: 2, lower bound: -0.0006276, upper bound: 0.0006276

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0052375, 0.0052394
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023392, 0.0023390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006244, upper bound: 0.0006175
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006165, upper bound: 0.0006244
time: 0.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0052394, 0.0052375
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023390, 0.0023392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006244, upper bound: 0.0006165
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006174, upper bound: 0.0006244
time: 1.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 2, lower bound: -0.0006244, upper bound: 0.0006175
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 2, lower bound: -0.0006165, upper bound: 0.0006244
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 2, lower bound: -0.0006244, upper bound: 0.0006165
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 2, lower bound: -0.0006174, upper bound: 0.0006244

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050552, 0.0049700
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023109, 0.0023203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006181, upper bound: 0.0006168
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006237, upper bound: 0.0006108
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049682, 0.0050551
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023202, 0.0023107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006107, upper bound: 0.0006237
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006158, upper bound: 0.0006179
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050551, 0.0049682
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023107, 0.0023202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006180, upper bound: 0.0006158
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006236, upper bound: 0.0006107
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049700, 0.0050552
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023203, 0.0023109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006108, upper bound: 0.0006237
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006168, upper bound: 0.0006181
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 2, lower bound: -0.0006181, upper bound: 0.0006168
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 2, lower bound: -0.0006237, upper bound: 0.0006108
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 2, lower bound: -0.0006107, upper bound: 0.0006237
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 2, lower bound: -0.0006158, upper bound: 0.0006179
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 2, lower bound: -0.0006180, upper bound: 0.0006158
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 2, lower bound: -0.0006236, upper bound: 0.0006107
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 2, lower bound: -0.0006108, upper bound: 0.0006237
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 2, lower bound: -0.0006168, upper bound: 0.0006181

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049295, 0.0048827
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023024, 0.0023076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006169, upper bound: 0.0006153
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006166, upper bound: 0.0006154
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049661, 0.0048442
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022982, 0.0023116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006222, upper bound: 0.0006092
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006222, upper bound: 0.0006095
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048424, 0.0049661
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023116, 0.0022980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006094, upper bound: 0.0006222
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006091, upper bound: 0.0006223
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048804, 0.0049293
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023075, 0.0023022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006145, upper bound: 0.0006164
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006142, upper bound: 0.0006167
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049293, 0.0048804
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023022, 0.0023075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006166, upper bound: 0.0006142
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006164, upper bound: 0.0006145
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049661, 0.0048424
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022980, 0.0023116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006223, upper bound: 0.0006091
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006222, upper bound: 0.0006094
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048442, 0.0049661
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023116, 0.0022982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006095, upper bound: 0.0006222
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006092, upper bound: 0.0006222
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048827, 0.0049295
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023076, 0.0023024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006153, upper bound: 0.0006166
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006153, upper bound: 0.0006169
time: 0.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006169, upper bound: 0.0006153
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006166, upper bound: 0.0006154
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006222, upper bound: 0.0006092
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006222, upper bound: 0.0006095
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006094, upper bound: 0.0006222
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006091, upper bound: 0.0006223
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006145, upper bound: 0.0006164
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006142, upper bound: 0.0006167
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006166, upper bound: 0.0006142
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006164, upper bound: 0.0006145
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006223, upper bound: 0.0006091
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006222, upper bound: 0.0006094
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006095, upper bound: 0.0006222
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006092, upper bound: 0.0006222
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006153, upper bound: 0.0006166
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.92
Output dim: 2, lower bound: -0.0006153, upper bound: 0.0006169

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049133, 0.0048587
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022997, 0.0023057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006037, upper bound: 0.0006149
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006165, upper bound: 0.0006041
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049055, 0.0048827
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023024, 0.0023049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006041, upper bound: 0.0006150
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006162, upper bound: 0.0006038
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049494, 0.0048202
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022955, 0.0023097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0006088
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006219, upper bound: 0.0006010
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049421, 0.0048442
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022982, 0.0023089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006062, upper bound: 0.0006091
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006218, upper bound: 0.0006008
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048245, 0.0049421
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023089, 0.0022960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006002, upper bound: 0.0006218
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006090, upper bound: 0.0006069
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048184, 0.0049661
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023116, 0.0022953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006003, upper bound: 0.0006219
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006087, upper bound: 0.0006063
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048634, 0.0049053
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023048, 0.0023002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006025, upper bound: 0.0006160
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006141, upper bound: 0.0006042
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048564, 0.0049293
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023075, 0.0022995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006026, upper bound: 0.0006163
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006139, upper bound: 0.0006039
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049125, 0.0048564
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022995, 0.0023056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006039, upper bound: 0.0006139
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006163, upper bound: 0.0006026
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049053, 0.0048804
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023022, 0.0023049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0006142
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006160, upper bound: 0.0006025
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049489, 0.0048184
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022953, 0.0023096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006063, upper bound: 0.0006087
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006219, upper bound: 0.0006002
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049421, 0.0048424
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022980, 0.0023089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006069, upper bound: 0.0006090
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006218, upper bound: 0.0006002
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048257, 0.0049421
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023089, 0.0022961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006008, upper bound: 0.0006218
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006091, upper bound: 0.0006062
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048202, 0.0049661
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023116, 0.0022955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006010, upper bound: 0.0006218
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006088, upper bound: 0.0006056
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048652, 0.0049055
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023049, 0.0023004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006039, upper bound: 0.0006162
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006150, upper bound: 0.0006041
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0048587, 0.0049295
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023076, 0.0022997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006041, upper bound: 0.0006165
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006149, upper bound: 0.0006037
time: 0.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006037, upper bound: 0.0006149
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006165, upper bound: 0.0006041
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006041, upper bound: 0.0006150
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006162, upper bound: 0.0006038
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0006088
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006219, upper bound: 0.0006010
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006062, upper bound: 0.0006091
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006218, upper bound: 0.0006008
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006002, upper bound: 0.0006218
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006090, upper bound: 0.0006069
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006003, upper bound: 0.0006219
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006087, upper bound: 0.0006063
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006025, upper bound: 0.0006160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006141, upper bound: 0.0006042
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006026, upper bound: 0.0006163
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006139, upper bound: 0.0006039
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006039, upper bound: 0.0006139
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006163, upper bound: 0.0006026
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0006142
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006160, upper bound: 0.0006025
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006063, upper bound: 0.0006087
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006219, upper bound: 0.0006002
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006069, upper bound: 0.0006090
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006218, upper bound: 0.0006002
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006008, upper bound: 0.0006218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006091, upper bound: 0.0006062
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006010, upper bound: 0.0006218
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006088, upper bound: 0.0006056
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006039, upper bound: 0.0006162
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006150, upper bound: 0.0006041
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006041, upper bound: 0.0006165
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 2, lower bound: -0.0006149, upper bound: 0.0006037

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050349, 0.0050795
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023249, 0.0023201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005899, upper bound: 0.0006118
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005999, upper bound: 0.0005908
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0051341, 0.0049979
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023160, 0.0023309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005947, upper bound: 0.0006004
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006133, upper bound: 0.0005875
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050323, 0.0051045
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023276, 0.0023198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005899, upper bound: 0.0006119
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006003, upper bound: 0.0005908
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0051263, 0.0050135
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023177, 0.0023301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005943, upper bound: 0.0006002
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006131, upper bound: 0.0005876
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050702, 0.0050410
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023207, 0.0023239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0006056
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006019, upper bound: 0.0005895
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0051702, 0.0049589
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023117, 0.0023349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005958, upper bound: 0.0005972
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006188, upper bound: 0.0005868
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050689, 0.0050659
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023234, 0.0023238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0006059
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006025, upper bound: 0.0005897
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0051629, 0.0049745
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023134, 0.0023341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005955, upper bound: 0.0005970
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006187, upper bound: 0.0005868
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049588, 0.0051629
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023341, 0.0023117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005862, upper bound: 0.0006188
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005964, upper bound: 0.0005958
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050453, 0.0050690
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023238, 0.0023212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005894, upper bound: 0.0006033
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006057, upper bound: 0.0005911
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049557, 0.0051878
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023368, 0.0023114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005862, upper bound: 0.0006188
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0005960
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050392, 0.0050846
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023255, 0.0023205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005892, upper bound: 0.0006026
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006055, upper bound: 0.0005911
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049969, 0.0051261
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023301, 0.0023159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005870, upper bound: 0.0006129
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005988, upper bound: 0.0005943
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050842, 0.0050325
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023198, 0.0023255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005903, upper bound: 0.0006006
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006110, upper bound: 0.0005900
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049946, 0.0051511
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023328, 0.0023156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005869, upper bound: 0.0006131
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005990, upper bound: 0.0005947
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050772, 0.0050481
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023215, 0.0023247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005903, upper bound: 0.0006001
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006107, upper bound: 0.0005900
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050345, 0.0050772
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023247, 0.0023200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005900, upper bound: 0.0006107
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006001, upper bound: 0.0005903
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0051333, 0.0049946
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023156, 0.0023308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005947, upper bound: 0.0005990
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006131, upper bound: 0.0005870
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050325, 0.0051022
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023274, 0.0023198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005900, upper bound: 0.0006110
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006006, upper bound: 0.0005903
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0051261, 0.0050102
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023173, 0.0023301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005943, upper bound: 0.0005988
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006129, upper bound: 0.0005870
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050707, 0.0050392
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023205, 0.0023240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005911, upper bound: 0.0006055
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006026, upper bound: 0.0005892
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0051697, 0.0049557
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023114, 0.0023348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005960, upper bound: 0.0005965
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006188, upper bound: 0.0005862
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050690, 0.0050641
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023232, 0.0023238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005911, upper bound: 0.0006057
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006033, upper bound: 0.0005894
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0051629, 0.0049713
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023131, 0.0023341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005958, upper bound: 0.0005964
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006188, upper bound: 0.0005863
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049616, 0.0051629
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023341, 0.0023120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005868, upper bound: 0.0006187
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0005955
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050465, 0.0050689
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023238, 0.0023213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005897, upper bound: 0.0006025
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006059, upper bound: 0.0005908
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049589, 0.0051879
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023368, 0.0023117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005868, upper bound: 0.0006188
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005972, upper bound: 0.0005957
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050410, 0.0050845
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023255, 0.0023207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0006019
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0005908
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049999, 0.0051263
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023301, 0.0023162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005875, upper bound: 0.0006131
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006002, upper bound: 0.0005943
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050860, 0.0050323
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023198, 0.0023257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0006003
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006119, upper bound: 0.0005899
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0049979, 0.0051512
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023328, 0.0023160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005875, upper bound: 0.0006133
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006004, upper bound: 0.0005947
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0050795, 0.0050479
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0023215, 0.0023249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0005999
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006118, upper bound: 0.0005899
time: 1.07 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005899, upper bound: 0.0006118
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005999, upper bound: 0.0005908
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005947, upper bound: 0.0006004
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006133, upper bound: 0.0005875
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005899, upper bound: 0.0006119
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006003, upper bound: 0.0005908
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005943, upper bound: 0.0006002
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006131, upper bound: 0.0005876
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0006056
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006019, upper bound: 0.0005895
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005958, upper bound: 0.0005972
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006188, upper bound: 0.0005868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0006059
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006025, upper bound: 0.0005897
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005955, upper bound: 0.0005970
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006187, upper bound: 0.0005868
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005862, upper bound: 0.0006188
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005964, upper bound: 0.0005958
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005894, upper bound: 0.0006033
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006057, upper bound: 0.0005911
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005862, upper bound: 0.0006188
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0005960
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005892, upper bound: 0.0006026
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006055, upper bound: 0.0005911
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005870, upper bound: 0.0006129
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005988, upper bound: 0.0005943
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005903, upper bound: 0.0006006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006110, upper bound: 0.0005900
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005869, upper bound: 0.0006131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005990, upper bound: 0.0005947
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005903, upper bound: 0.0006001
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006107, upper bound: 0.0005900
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005900, upper bound: 0.0006107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006001, upper bound: 0.0005903
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005947, upper bound: 0.0005990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006131, upper bound: 0.0005870
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005900, upper bound: 0.0006110
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006006, upper bound: 0.0005903
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005943, upper bound: 0.0005988
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006129, upper bound: 0.0005870
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005911, upper bound: 0.0006055
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006026, upper bound: 0.0005892
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005960, upper bound: 0.0005965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006188, upper bound: 0.0005862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005911, upper bound: 0.0006057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006033, upper bound: 0.0005894
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005958, upper bound: 0.0005964
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006188, upper bound: 0.0005863
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005868, upper bound: 0.0006187
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0005955
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005897, upper bound: 0.0006025
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006059, upper bound: 0.0005908
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005868, upper bound: 0.0006188
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005972, upper bound: 0.0005957
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0006019
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0005908
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005875, upper bound: 0.0006131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006002, upper bound: 0.0005943
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0006003
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006119, upper bound: 0.0005899
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005875, upper bound: 0.0006133
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006004, upper bound: 0.0005947
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0005999
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.02
Output dim: 2, lower bound: -0.0006118, upper bound: 0.0005899

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046761, 0.0047044
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022833, 0.0022802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005871, upper bound: 0.0006058
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005894, upper bound: 0.0006115
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047531, 0.0046216
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022742, 0.0022886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005992, upper bound: 0.0005902
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005994, upper bound: 0.0005904
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046761, 0.0047044
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022833, 0.0022802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005943, upper bound: 0.0006000
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005937, upper bound: 0.0006000
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047531, 0.0046216
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022742, 0.0022886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006130, upper bound: 0.0005870
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006060, upper bound: 0.0005850
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046683, 0.0047265
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022857, 0.0022793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005871, upper bound: 0.0006057
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005894, upper bound: 0.0006115
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047466, 0.0046467
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022769, 0.0022879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005998, upper bound: 0.0005902
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005999, upper bound: 0.0005904
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046683, 0.0047265
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022857, 0.0022793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005938, upper bound: 0.0005998
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005934, upper bound: 0.0005997
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047466, 0.0046467
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022769, 0.0022879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006128, upper bound: 0.0005870
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006060, upper bound: 0.0005849
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047122, 0.0046625
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022787, 0.0022841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005877, upper bound: 0.0006006
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005903, upper bound: 0.0006053
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047952, 0.0045830
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022699, 0.0022932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006013, upper bound: 0.0005889
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006015, upper bound: 0.0005891
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047122, 0.0046625
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022787, 0.0022841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005954, upper bound: 0.0005968
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005948, upper bound: 0.0005967
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047952, 0.0045830
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022699, 0.0022932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006185, upper bound: 0.0005863
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006100, upper bound: 0.0005843
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047050, 0.0046846
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022811, 0.0022833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005877, upper bound: 0.0006006
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005903, upper bound: 0.0006055
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047881, 0.0046082
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022727, 0.0022925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006020, upper bound: 0.0005890
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006021, upper bound: 0.0005893
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047050, 0.0046846
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022811, 0.0022833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005951, upper bound: 0.0005966
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005947, upper bound: 0.0005965
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047881, 0.0046082
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022727, 0.0022925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006184, upper bound: 0.0005862
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006100, upper bound: 0.0005843
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0045874, 0.0047875
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022924, 0.0022704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005840, upper bound: 0.0006109
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005857, upper bound: 0.0006185
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046670, 0.0047049
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022833, 0.0022792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005959, upper bound: 0.0005949
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005959, upper bound: 0.0005954
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0045874, 0.0047875
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022924, 0.0022704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005890, upper bound: 0.0006028
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005885, upper bound: 0.0006026
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046670, 0.0047049
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022833, 0.0022792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006054, upper bound: 0.0005906
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005998, upper bound: 0.0005879
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0045812, 0.0048096
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022948, 0.0022697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005840, upper bound: 0.0006107
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005857, upper bound: 0.0006185
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046601, 0.0047301
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022860, 0.0022784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005961, upper bound: 0.0005951
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005961, upper bound: 0.0005956
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0045812, 0.0048096
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022948, 0.0022697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005888, upper bound: 0.0006021
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005884, upper bound: 0.0006018
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046601, 0.0047301
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022860, 0.0022784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006052, upper bound: 0.0005906
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005998, upper bound: 0.0005879
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046262, 0.0047456
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022878, 0.0022747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005846, upper bound: 0.0006061
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005865, upper bound: 0.0006126
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047101, 0.0046682
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022793, 0.0022839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005983, upper bound: 0.0005934
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005983, upper bound: 0.0005939
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046262, 0.0047456
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022878, 0.0022747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005900, upper bound: 0.0006001
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0006000
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047101, 0.0046682
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022793, 0.0022839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006106, upper bound: 0.0005895
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0005872
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046192, 0.0047677
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022902, 0.0022739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005847, upper bound: 0.0006061
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005865, upper bound: 0.0006128
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047023, 0.0046933
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022820, 0.0022830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005986, upper bound: 0.0005938
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005986, upper bound: 0.0005943
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046192, 0.0047677
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022902, 0.0022739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005899, upper bound: 0.0005997
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0005993
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047023, 0.0046933
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022820, 0.0022830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006104, upper bound: 0.0005895
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0005872
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046753, 0.0047023
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022830, 0.0022801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005872, upper bound: 0.0006044
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0006104
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047511, 0.0046192
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022739, 0.0022884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005993, upper bound: 0.0005895
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005997, upper bound: 0.0005899
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046753, 0.0047023
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022830, 0.0022801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005943, upper bound: 0.0005986
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005938, upper bound: 0.0005986
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047511, 0.0046192
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022739, 0.0022884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006127, upper bound: 0.0005865
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006061, upper bound: 0.0005847
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046682, 0.0047244
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022854, 0.0022793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005872, upper bound: 0.0006044
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005894, upper bound: 0.0006107
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0047456, 0.0046444
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022766, 0.0022878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005999, upper bound: 0.0005895
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006001, upper bound: 0.0005899
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041320, -0.0042018, -0.0041320, -0.0000698, 0.0000698
1: -0.0100127, -0.0092445, -0.0100127, -0.0092445, -0.0007682, 0.0007682
2: 0.9644478, 0.9653697, 0.9644478, 0.9653697, -0.0009219, 0.0009219
3: -0.0159214, -0.0091215, -0.0159214, -0.0091215, -0.0046682, 0.0047244
4: 0.0000007, 0.0005179, 0.0000007, 0.0005179, -0.0005172, 0.0005172
5: 0.0172711, 0.0180427, 0.0172711, 0.0180427, -0.0007716, 0.0007716
6: 0.0025888, 0.0035061, 0.0025888, 0.0035061, -0.0009173, 0.0009173
7: -0.0054143, -0.0032583, -0.0054143, -0.0032583, -0.0021560, 0.0021560
8: 0.0124337, 0.0138317, 0.0124337, 0.0138317, -0.0013981, 0.0013981
9: 0.0200878, 0.0226023, 0.0200878, 0.0226023, -0.0022854, 0.0022793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0005983
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005934, upper bound: 0.0005983
time: 1.00 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 7.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005871, upper bound: 0.0006058
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005894, upper bound: 0.0006115
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005992, upper bound: 0.0005902
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005994, upper bound: 0.0005904
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005943, upper bound: 0.0006000
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005937, upper bound: 0.0006000
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006130, upper bound: 0.0005870
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006060, upper bound: 0.0005850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005871, upper bound: 0.0006057
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005894, upper bound: 0.0006115
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005998, upper bound: 0.0005902
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005999, upper bound: 0.0005904
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005938, upper bound: 0.0005998
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005934, upper bound: 0.0005997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006128, upper bound: 0.0005870
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006060, upper bound: 0.0005849
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005877, upper bound: 0.0006006
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005903, upper bound: 0.0006053
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006013, upper bound: 0.0005889
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006015, upper bound: 0.0005891
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005954, upper bound: 0.0005968
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005948, upper bound: 0.0005967
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006185, upper bound: 0.0005863
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006100, upper bound: 0.0005843
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005877, upper bound: 0.0006006
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005903, upper bound: 0.0006055
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006020, upper bound: 0.0005890
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006021, upper bound: 0.0005893
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005951, upper bound: 0.0005966
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005947, upper bound: 0.0005965
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006184, upper bound: 0.0005862
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006100, upper bound: 0.0005843
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005840, upper bound: 0.0006109
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005857, upper bound: 0.0006185
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005959, upper bound: 0.0005949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005959, upper bound: 0.0005954
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005890, upper bound: 0.0006028
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005885, upper bound: 0.0006026
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006054, upper bound: 0.0005906
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005998, upper bound: 0.0005879
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005840, upper bound: 0.0006107
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005857, upper bound: 0.0006185
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005961, upper bound: 0.0005951
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005961, upper bound: 0.0005956
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005888, upper bound: 0.0006021
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005884, upper bound: 0.0006018
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006052, upper bound: 0.0005906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005998, upper bound: 0.0005879
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005846, upper bound: 0.0006061
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005865, upper bound: 0.0006126
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005983, upper bound: 0.0005934
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005983, upper bound: 0.0005939
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005900, upper bound: 0.0006001
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0006000
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006106, upper bound: 0.0005895
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0005872
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005847, upper bound: 0.0006061
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005865, upper bound: 0.0006128
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005986, upper bound: 0.0005938
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005986, upper bound: 0.0005943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005899, upper bound: 0.0005997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0005993
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006104, upper bound: 0.0005895
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0005872
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005872, upper bound: 0.0006044
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0006104
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005993, upper bound: 0.0005895
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005997, upper bound: 0.0005899
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005943, upper bound: 0.0005986
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005938, upper bound: 0.0005986
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006127, upper bound: 0.0005865
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006061, upper bound: 0.0005847
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005872, upper bound: 0.0006044
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005894, upper bound: 0.0006107
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005999, upper bound: 0.0005895
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0006001, upper bound: 0.0005899
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0005983
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.10
Output dim: 2, lower bound: -0.0005934, upper bound: 0.0005983
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006129, upper bound: 0.0005870
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005911, upper bound: 0.0006055
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006026, upper bound: 0.0005892
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005960, upper bound: 0.0005965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006188, upper bound: 0.0005862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005911, upper bound: 0.0006057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006033, upper bound: 0.0005894
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005958, upper bound: 0.0005964
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006188, upper bound: 0.0005863
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005868, upper bound: 0.0006187
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0005955
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005897, upper bound: 0.0006025
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006059, upper bound: 0.0005908
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005868, upper bound: 0.0006188
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005972, upper bound: 0.0005957
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005895, upper bound: 0.0006019
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0005908
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005875, upper bound: 0.0006131
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006002, upper bound: 0.0005943
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0006003
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006119, upper bound: 0.0005899
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005875, upper bound: 0.0006133
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006004, upper bound: 0.0005947
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0005908, upper bound: 0.0005999
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.10
Output dim: 2, lower bound: -0.0006118, upper bound: 0.0005899

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.87 + 597.56 = 600.43 seconds
