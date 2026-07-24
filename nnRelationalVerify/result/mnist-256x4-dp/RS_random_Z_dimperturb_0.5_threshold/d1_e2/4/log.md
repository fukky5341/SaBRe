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
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704)
1: (-0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790)
2: (0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347)
3: (-0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052669, 0.0052669)
4: (-0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244)
5: (0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811)
6: (0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266)
7: (-0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841)
8: (0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176)
9: (0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023671, 0.0023671)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.13 + 1.65 = 2.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0006363, upper bound: 0.0006363

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006331, upper bound: 0.0006301
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006300, upper bound: 0.0006331
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 2, lower bound: -0.0006331, upper bound: 0.0006301
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 2, lower bound: -0.0006300, upper bound: 0.0006331

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051199, 0.0050619
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023438, 0.0023502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006266, upper bound: 0.0006290
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006320, upper bound: 0.0006229
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050619, 0.0051199
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023502, 0.0023438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006229, upper bound: 0.0006321
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006290, upper bound: 0.0006266
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 2, lower bound: -0.0006266, upper bound: 0.0006290
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 2, lower bound: -0.0006320, upper bound: 0.0006229
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 2, lower bound: -0.0006229, upper bound: 0.0006321
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 2, lower bound: -0.0006290, upper bound: 0.0006266

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049749, 0.0049552
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023343, 0.0023365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006251, upper bound: 0.0006273
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006250, upper bound: 0.0006273
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050099, 0.0049169
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023301, 0.0023403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006313, upper bound: 0.0006221
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006312, upper bound: 0.0006222
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049169, 0.0050099
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023403, 0.0023301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006214, upper bound: 0.0006302
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006212, upper bound: 0.0006301
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049552, 0.0049749
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023365, 0.0023343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006281, upper bound: 0.0006253
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006283, upper bound: 0.0006259
time: 0.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 2, lower bound: -0.0006251, upper bound: 0.0006273
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 2, lower bound: -0.0006250, upper bound: 0.0006273
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 2, lower bound: -0.0006313, upper bound: 0.0006221
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 2, lower bound: -0.0006312, upper bound: 0.0006222
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 2, lower bound: -0.0006214, upper bound: 0.0006302
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 2, lower bound: -0.0006212, upper bound: 0.0006301
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 2, lower bound: -0.0006281, upper bound: 0.0006253
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 2, lower bound: -0.0006283, upper bound: 0.0006259

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049497, 0.0049297
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023317, 0.0023339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006244, upper bound: 0.0006266
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006238, upper bound: 0.0006264
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049495, 0.0049552
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023343, 0.0023339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006242, upper bound: 0.0006266
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006236, upper bound: 0.0006264
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050074, 0.0049139
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023301, 0.0023404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006294, upper bound: 0.0006205
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006295, upper bound: 0.0006206
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050054, 0.0049145
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023302, 0.0023402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006293, upper bound: 0.0006205
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006294, upper bound: 0.0006207
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048932, 0.0049844
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023377, 0.0023277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006207, upper bound: 0.0006294
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006206, upper bound: 0.0006295
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048915, 0.0050099
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023403, 0.0023275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006205, upper bound: 0.0006293
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006205, upper bound: 0.0006294
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049527, 0.0049701
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023363, 0.0023344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006264, upper bound: 0.0006236
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006264, upper bound: 0.0006238
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049535, 0.0049724
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023365, 0.0023345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006266, upper bound: 0.0006242
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006266, upper bound: 0.0006244
time: 0.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006244, upper bound: 0.0006266
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006238, upper bound: 0.0006264
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006242, upper bound: 0.0006266
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006236, upper bound: 0.0006264
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006294, upper bound: 0.0006205
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006295, upper bound: 0.0006206
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006293, upper bound: 0.0006205
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006294, upper bound: 0.0006207
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006207, upper bound: 0.0006294
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006206, upper bound: 0.0006295
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006205, upper bound: 0.0006293
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006205, upper bound: 0.0006294
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006264, upper bound: 0.0006236
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006264, upper bound: 0.0006238
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006266, upper bound: 0.0006242
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 2, lower bound: -0.0006266, upper bound: 0.0006244

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049474, 0.0049282
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023319, 0.0023340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006084, upper bound: 0.0006264
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006241, upper bound: 0.0006106
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049451, 0.0049274
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023318, 0.0023337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006084, upper bound: 0.0006262
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006236, upper bound: 0.0006101
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049472, 0.0049535
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023345, 0.0023339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006013, upper bound: 0.0006233
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006210, upper bound: 0.0006002
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049448, 0.0049527
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023344, 0.0023337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006090, upper bound: 0.0006262
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006234, upper bound: 0.0006095
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049803, 0.0048886
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023275, 0.0023376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006291, upper bound: 0.0006138
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006198, upper bound: 0.0006202
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049821, 0.0049139
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023301, 0.0023378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006030, upper bound: 0.0006173
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006262, upper bound: 0.0005982
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049782, 0.0048892
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023276, 0.0023374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006101, upper bound: 0.0006203
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006291, upper bound: 0.0006079
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049801, 0.0049145
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023302, 0.0023376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006291, upper bound: 0.0006134
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006207, upper bound: 0.0006204
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048909, 0.0049801
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023376, 0.0023278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005982, upper bound: 0.0006262
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006173, upper bound: 0.0006030
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048901, 0.0049821
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023378, 0.0023277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005982, upper bound: 0.0006262
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006173, upper bound: 0.0006030
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048892, 0.0050054
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023402, 0.0023276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006202, upper bound: 0.0006203
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006134, upper bound: 0.0006291
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048886, 0.0050074
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023404, 0.0023275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006081, upper bound: 0.0006292
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006203, upper bound: 0.0006099
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049259, 0.0049448
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023337, 0.0023316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006002, upper bound: 0.0006204
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006232, upper bound: 0.0006011
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049274, 0.0049701
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023363, 0.0023318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006261, upper bound: 0.0006157
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006177, upper bound: 0.0006235
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049264, 0.0049472
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023339, 0.0023317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006263, upper bound: 0.0006156
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006186, upper bound: 0.0006239
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049282, 0.0049724
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023365, 0.0023319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006263, upper bound: 0.0006156
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006187, upper bound: 0.0006241
time: 0.97 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006084, upper bound: 0.0006264
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006241, upper bound: 0.0006106
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006084, upper bound: 0.0006262
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006236, upper bound: 0.0006101
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006013, upper bound: 0.0006233
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006210, upper bound: 0.0006002
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006090, upper bound: 0.0006262
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006234, upper bound: 0.0006095
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006291, upper bound: 0.0006138
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006198, upper bound: 0.0006202
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006030, upper bound: 0.0006173
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006262, upper bound: 0.0005982
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006101, upper bound: 0.0006203
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006291, upper bound: 0.0006079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006291, upper bound: 0.0006134
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006207, upper bound: 0.0006204
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0005982, upper bound: 0.0006262
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006173, upper bound: 0.0006030
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0005982, upper bound: 0.0006262
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006173, upper bound: 0.0006030
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006202, upper bound: 0.0006203
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006134, upper bound: 0.0006291
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006081, upper bound: 0.0006292
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006203, upper bound: 0.0006099
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006002, upper bound: 0.0006204
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006232, upper bound: 0.0006011
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006261, upper bound: 0.0006157
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006177, upper bound: 0.0006235
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006263, upper bound: 0.0006156
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006186, upper bound: 0.0006239
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006263, upper bound: 0.0006156
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.84
Output dim: 2, lower bound: -0.0006187, upper bound: 0.0006241

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051042, 0.0051861
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023610, 0.0023520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0006231
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006042, upper bound: 0.0005999
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052052, 0.0050953
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023511, 0.0023631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006239, upper bound: 0.0006101
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006154, upper bound: 0.0006100
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051015, 0.0051853
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023609, 0.0023517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006078, upper bound: 0.0006175
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006080, upper bound: 0.0006259
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052029, 0.0050946
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023510, 0.0023629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006233, upper bound: 0.0006096
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006155, upper bound: 0.0006096
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047089, 0.0047934
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023168, 0.0023078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0006231
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006011, upper bound: 0.0006056
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047925, 0.0047158
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023083, 0.0023170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006207, upper bound: 0.0005997
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006114, upper bound: 0.0005998
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051028, 0.0052097
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023635, 0.0023519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005966, upper bound: 0.0006230
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006049, upper bound: 0.0006000
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052027, 0.0051098
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023526, 0.0023628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006231, upper bound: 0.0006090
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006155, upper bound: 0.0006089
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049699, 0.0048735
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023257, 0.0023363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006093, upper bound: 0.0006136
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006289, upper bound: 0.0006076
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049652, 0.0048779
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023262, 0.0023358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006026, upper bound: 0.0006169
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006158, upper bound: 0.0005975
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047439, 0.0047534
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023125, 0.0023116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005975, upper bound: 0.0006171
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006028, upper bound: 0.0006036
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048298, 0.0046762
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023039, 0.0023211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006068, upper bound: 0.0005981
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006260, upper bound: 0.0005946
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051358, 0.0051471
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023567, 0.0023555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006094, upper bound: 0.0006132
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006097, upper bound: 0.0006200
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052361, 0.0050587
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023470, 0.0023665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006288, upper bound: 0.0006075
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006201, upper bound: 0.0006074
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049697, 0.0048992
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023284, 0.0023363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006106, upper bound: 0.0006132
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006289, upper bound: 0.0006072
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049650, 0.0049041
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023289, 0.0023358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006023, upper bound: 0.0006170
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006168, upper bound: 0.0005978
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0046527, 0.0048262
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023207, 0.0023016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005946, upper bound: 0.0006260
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005981, upper bound: 0.0006072
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047365, 0.0047419
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023114, 0.0023108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006034, upper bound: 0.0006028
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006171, upper bound: 0.0005975
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0046519, 0.0048297
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023211, 0.0023015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005978, upper bound: 0.0006163
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005978, upper bound: 0.0006260
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047330, 0.0047439
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023116, 0.0023104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006170, upper bound: 0.0006023
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006098, upper bound: 0.0006026
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048786, 0.0049902
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023384, 0.0023263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005975, upper bound: 0.0006162
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006169, upper bound: 0.0006026
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048740, 0.0049952
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023389, 0.0023258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006075, upper bound: 0.0006288
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006132, upper bound: 0.0006094
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050593, 0.0052644
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023695, 0.0023471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006076, upper bound: 0.0006197
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006076, upper bound: 0.0006289
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051465, 0.0051548
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023575, 0.0023567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005978, upper bound: 0.0006058
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006170, upper bound: 0.0005974
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0046877, 0.0047881
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023165, 0.0023055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005955, upper bound: 0.0006202
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006000, upper bound: 0.0006049
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047722, 0.0047066
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023075, 0.0023147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006229, upper bound: 0.0006005
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006137, upper bound: 0.0006007
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049166, 0.0049549
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023345, 0.0023305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005996, upper bound: 0.0006114
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006229, upper bound: 0.0006010
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049123, 0.0049600
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023350, 0.0023300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006202
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006138, upper bound: 0.0006013
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049157, 0.0049320
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023321, 0.0023304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006090, upper bound: 0.0006154
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006261, upper bound: 0.0006084
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049113, 0.0049367
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023327, 0.0023299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006093, upper bound: 0.0006237
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006184, upper bound: 0.0006084
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049174, 0.0049572
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023347, 0.0023305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006100, upper bound: 0.0006154
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006261, upper bound: 0.0006080
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049130, 0.0049622
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023353, 0.0023301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

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
Output dim: 2, lower bound: -0.0005996, upper bound: 0.0006208
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006149, upper bound: 0.0006014
time: 0.96 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0006231
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006042, upper bound: 0.0005999
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006239, upper bound: 0.0006101
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006154, upper bound: 0.0006100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006078, upper bound: 0.0006175
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006080, upper bound: 0.0006259
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006233, upper bound: 0.0006096
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006155, upper bound: 0.0006096
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005965, upper bound: 0.0006231
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006011, upper bound: 0.0006056
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006207, upper bound: 0.0005997
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006114, upper bound: 0.0005998
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005966, upper bound: 0.0006230
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006049, upper bound: 0.0006000
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006231, upper bound: 0.0006090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006155, upper bound: 0.0006089
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006093, upper bound: 0.0006136
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006289, upper bound: 0.0006076
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006026, upper bound: 0.0006169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006158, upper bound: 0.0005975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005975, upper bound: 0.0006171
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006028, upper bound: 0.0006036
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006068, upper bound: 0.0005981
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006260, upper bound: 0.0005946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006094, upper bound: 0.0006132
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006097, upper bound: 0.0006200
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006288, upper bound: 0.0006075
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006201, upper bound: 0.0006074
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006106, upper bound: 0.0006132
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006289, upper bound: 0.0006072
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006023, upper bound: 0.0006170
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006168, upper bound: 0.0005978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005946, upper bound: 0.0006260
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005981, upper bound: 0.0006072
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006034, upper bound: 0.0006028
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006171, upper bound: 0.0005975
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005978, upper bound: 0.0006163
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005978, upper bound: 0.0006260
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006170, upper bound: 0.0006023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006098, upper bound: 0.0006026
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005975, upper bound: 0.0006162
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006169, upper bound: 0.0006026
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006075, upper bound: 0.0006288
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006132, upper bound: 0.0006094
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006076, upper bound: 0.0006197
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006076, upper bound: 0.0006289
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005978, upper bound: 0.0006058
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006170, upper bound: 0.0005974
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005955, upper bound: 0.0006202
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006000, upper bound: 0.0006049
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006229, upper bound: 0.0006005
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006137, upper bound: 0.0006007
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005996, upper bound: 0.0006114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006229, upper bound: 0.0006010
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006202
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006138, upper bound: 0.0006013
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006090, upper bound: 0.0006154
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006261, upper bound: 0.0006084
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006093, upper bound: 0.0006237
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006184, upper bound: 0.0006084
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006100, upper bound: 0.0006154
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006261, upper bound: 0.0006080
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0005996, upper bound: 0.0006208
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.08
Output dim: 2, lower bound: -0.0006149, upper bound: 0.0006014

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047091, 0.0047707
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023146, 0.0023078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006147
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005960, upper bound: 0.0006228
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047934, 0.0046900
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023057, 0.0023171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006036, upper bound: 0.0005994
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006037, upper bound: 0.0005995
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049373, 0.0049130
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023301, 0.0023327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006012, upper bound: 0.0006062
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006206, upper bound: 0.0005951
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049322, 0.0049174
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023305, 0.0023322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006008, upper bound: 0.0006061
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006111, upper bound: 0.0005931
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049350, 0.0049123
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023300, 0.0023325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005937, upper bound: 0.0006136
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006037, upper bound: 0.0005994
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049299, 0.0049166
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023305, 0.0023319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005961, upper bound: 0.0006227
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006038, upper bound: 0.0005995
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049350, 0.0049123
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023300, 0.0023325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006011, upper bound: 0.0006056
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006200, upper bound: 0.0005950
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049299, 0.0049166
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023305, 0.0023319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006008, upper bound: 0.0006056
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006113, upper bound: 0.0005931
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051059, 0.0052104
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023636, 0.0023522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006145
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005960, upper bound: 0.0006228
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052050, 0.0051106
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023527, 0.0023631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006008, upper bound: 0.0006052
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006003, upper bound: 0.0006049
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049367, 0.0049383
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023327, 0.0023327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0005995
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006205, upper bound: 0.0005950
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049320, 0.0049428
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023332, 0.0023321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0005996
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006112, upper bound: 0.0005931
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047066, 0.0047938
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023169, 0.0023075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005937, upper bound: 0.0006136
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005961, upper bound: 0.0006227
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047881, 0.0047150
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023082, 0.0023165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006044, upper bound: 0.0005995
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006045, upper bound: 0.0005996
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049345, 0.0049375
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023326, 0.0023324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006005, upper bound: 0.0006049
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006199, upper bound: 0.0005950
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049297, 0.0049421
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023331, 0.0023319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006003, upper bound: 0.0006047
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006114, upper bound: 0.0005931
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051375, 0.0051465
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023567, 0.0023557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005940, upper bound: 0.0006096
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006052, upper bound: 0.0005972
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052382, 0.0050593
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023471, 0.0023667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006027, upper bound: 0.0006035
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006257, upper bound: 0.0005939
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047421, 0.0047307
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023102, 0.0023114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006167
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006025, upper bound: 0.0006035
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048277, 0.0046504
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023014, 0.0023208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

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

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006053, upper bound: 0.0005973
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006156, upper bound: 0.0005926
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051396, 0.0051709
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023593, 0.0023559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005941, upper bound: 0.0006096
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0006168
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052400, 0.0050745
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023487, 0.0023669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006024, upper bound: 0.0006031
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006022, upper bound: 0.0006031
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051396, 0.0051709
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023593, 0.0023559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006063, upper bound: 0.0005976
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006064, upper bound: 0.0005976
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052400, 0.0050745
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023487, 0.0023669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006257, upper bound: 0.0005940
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006161, upper bound: 0.0005927
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049678, 0.0048740
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023258, 0.0023361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006092
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006054, upper bound: 0.0005972
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049631, 0.0048786
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023263, 0.0023356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0006167
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0005973
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049678, 0.0048740
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023258, 0.0023361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006027, upper bound: 0.0006034
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006256, upper bound: 0.0005939
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049631, 0.0048786
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023263, 0.0023356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006025, upper bound: 0.0006034
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006161, upper bound: 0.0005926
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051371, 0.0051714
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023593, 0.0023556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006092
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006067, upper bound: 0.0005976
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0052380, 0.0050739
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023487, 0.0023667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006024, upper bound: 0.0006030
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006257, upper bound: 0.0005940
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047419, 0.0047550
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023126, 0.0023114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0006169
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006022, upper bound: 0.0006030
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048262, 0.0046768
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023040, 0.0023207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006068, upper bound: 0.0005976
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006166, upper bound: 0.0005927
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050590, 0.0052380
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023667, 0.0023471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005927, upper bound: 0.0006166
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005940, upper bound: 0.0006257
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051488, 0.0051371
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023556, 0.0023569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005976, upper bound: 0.0006068
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005976, upper bound: 0.0006067
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050590, 0.0052380
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023667, 0.0023471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006029, upper bound: 0.0006022
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006030, upper bound: 0.0006024
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051488, 0.0051371
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023556, 0.0023569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006168, upper bound: 0.0005970
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006091, upper bound: 0.0005939
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048796, 0.0049670
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023360, 0.0023264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005926, upper bound: 0.0006162
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005976, upper bound: 0.0006064
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048749, 0.0049716
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023365, 0.0023259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005940, upper bound: 0.0006258
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005976, upper bound: 0.0006064
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048796, 0.0049670
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023360, 0.0023264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006031, upper bound: 0.0006022
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006168, upper bound: 0.0005970
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048749, 0.0049716
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023365, 0.0023259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006031, upper bound: 0.0006024
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006096, upper bound: 0.0005941
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0046510, 0.0048489
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023229, 0.0023014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005926, upper bound: 0.0006161
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005973, upper bound: 0.0006056
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047323, 0.0047677
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023140, 0.0023104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006034, upper bound: 0.0006025
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006167, upper bound: 0.0005970
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050587, 0.0052624
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023693, 0.0023470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006256
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006034, upper bound: 0.0006027
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051471, 0.0051524
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023573, 0.0023567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005972, upper bound: 0.0006054
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006092, upper bound: 0.0005939
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048779, 0.0049922
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023386, 0.0023262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005926, upper bound: 0.0006156
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006035, upper bound: 0.0006025
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0048735, 0.0049971
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023391, 0.0023257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006257
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006035, upper bound: 0.0006027
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0046504, 0.0048524
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023233, 0.0023014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005973, upper bound: 0.0006053
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005972, upper bound: 0.0006052
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047307, 0.0047697
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023142, 0.0023102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 111

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006167, upper bound: 0.0005969
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006096, upper bound: 0.0005940
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050941, 0.0052027
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023628, 0.0023509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005931, upper bound: 0.0006114
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005950, upper bound: 0.0006199
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051838, 0.0051028
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023519, 0.0023608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005996, upper bound: 0.0006045
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006044
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049153, 0.0049297
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023319, 0.0023303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006047, upper bound: 0.0006003
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006227, upper bound: 0.0005961
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049108, 0.0049345
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023324, 0.0023298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006049, upper bound: 0.0006005
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006136, upper bound: 0.0005938
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0046892, 0.0048108
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023188, 0.0023056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005932, upper bound: 0.0006113
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006038
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047711, 0.0047324
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023101, 0.0023146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0006008
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006227, upper bound: 0.0005961
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0046892, 0.0048108
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023188, 0.0023056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005950, upper bound: 0.0006200
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005994, upper bound: 0.0006037
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047711, 0.0047324
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023101, 0.0023146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0006011
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006136, upper bound: 0.0005938
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050952, 0.0052050
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023631, 0.0023510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005931, upper bound: 0.0006112
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006049, upper bound: 0.0006003
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051843, 0.0051059
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023522, 0.0023608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005996, upper bound: 0.0006043
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006228, upper bound: 0.0005960
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050952, 0.0052050
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023631, 0.0023510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005951, upper bound: 0.0006205
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006052, upper bound: 0.0006007
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051843, 0.0051059
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023522, 0.0023608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006043
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006145, upper bound: 0.0005939
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0050953, 0.0052294
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023657, 0.0023511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005931, upper bound: 0.0006111
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006061, upper bound: 0.0006008
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0051861, 0.0051211
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023538, 0.0023610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 187
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006037
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006228, upper bound: 0.0005961
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0046900, 0.0048151
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023192, 0.0023057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 157
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 157

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 245

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005950, upper bound: 0.0006206
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005994, upper bound: 0.0006036
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0047707, 0.0047348
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023104, 0.0023146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 141
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 245
type: RSZ, layer: 3, pos: 157

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006062, upper bound: 0.0006012
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0006147, upper bound: 0.0005939
time: 1.10 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006147
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005960, upper bound: 0.0006228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006036, upper bound: 0.0005994
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006037, upper bound: 0.0005995
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006012, upper bound: 0.0006062
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006206, upper bound: 0.0005951
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006008, upper bound: 0.0006061
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006111, upper bound: 0.0005931
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005937, upper bound: 0.0006136
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006037, upper bound: 0.0005994
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005961, upper bound: 0.0006227
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006038, upper bound: 0.0005995
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006011, upper bound: 0.0006056
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006200, upper bound: 0.0005950
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006008, upper bound: 0.0006056
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006113, upper bound: 0.0005931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006145
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005960, upper bound: 0.0006228
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006008, upper bound: 0.0006052
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006003, upper bound: 0.0006049
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0005995
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006205, upper bound: 0.0005950
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006043, upper bound: 0.0005996
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006112, upper bound: 0.0005931
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005937, upper bound: 0.0006136
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005961, upper bound: 0.0006227
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006044, upper bound: 0.0005995
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006045, upper bound: 0.0005996
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006005, upper bound: 0.0006049
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006199, upper bound: 0.0005950
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006003, upper bound: 0.0006047
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006114, upper bound: 0.0005931
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005940, upper bound: 0.0006096
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006052, upper bound: 0.0005972
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006027, upper bound: 0.0006035
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006257, upper bound: 0.0005939
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005969, upper bound: 0.0006167
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006025, upper bound: 0.0006035
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006053, upper bound: 0.0005973
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006156, upper bound: 0.0005926
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005941, upper bound: 0.0006096
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0006168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006024, upper bound: 0.0006031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006022, upper bound: 0.0006031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006063, upper bound: 0.0005976
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006064, upper bound: 0.0005976
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006257, upper bound: 0.0005940
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006161, upper bound: 0.0005927
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006092
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006054, upper bound: 0.0005972
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0006167
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0005973
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006027, upper bound: 0.0006034
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006256, upper bound: 0.0005939
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006025, upper bound: 0.0006034
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006161, upper bound: 0.0005926
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006092
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006067, upper bound: 0.0005976
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006024, upper bound: 0.0006030
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006257, upper bound: 0.0005940
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005970, upper bound: 0.0006169
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006022, upper bound: 0.0006030
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006068, upper bound: 0.0005976
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006166, upper bound: 0.0005927
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005927, upper bound: 0.0006166
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005940, upper bound: 0.0006257
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005976, upper bound: 0.0006068
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005976, upper bound: 0.0006067
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006029, upper bound: 0.0006022
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006030, upper bound: 0.0006024
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006168, upper bound: 0.0005970
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006091, upper bound: 0.0005939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005926, upper bound: 0.0006162
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005976, upper bound: 0.0006064
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005940, upper bound: 0.0006258
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005976, upper bound: 0.0006064
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006031, upper bound: 0.0006022
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006168, upper bound: 0.0005970
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006031, upper bound: 0.0006024
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006096, upper bound: 0.0005941
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005926, upper bound: 0.0006161
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005973, upper bound: 0.0006056
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006034, upper bound: 0.0006025
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006167, upper bound: 0.0005970
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006256
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006034, upper bound: 0.0006027
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005972, upper bound: 0.0006054
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006092, upper bound: 0.0005939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005926, upper bound: 0.0006156
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006035, upper bound: 0.0006025
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005939, upper bound: 0.0006257
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006035, upper bound: 0.0006027
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005973, upper bound: 0.0006053
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005972, upper bound: 0.0006052
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006167, upper bound: 0.0005969
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006096, upper bound: 0.0005940
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005931, upper bound: 0.0006114
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005950, upper bound: 0.0006199
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005996, upper bound: 0.0006045
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006044
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006047, upper bound: 0.0006003
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006227, upper bound: 0.0005961
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006049, upper bound: 0.0006005
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006136, upper bound: 0.0005938
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005932, upper bound: 0.0006113
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006038
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0006008
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006227, upper bound: 0.0005961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005950, upper bound: 0.0006200
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005994, upper bound: 0.0006037
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006056, upper bound: 0.0006011
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006136, upper bound: 0.0005938
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005931, upper bound: 0.0006112
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006049, upper bound: 0.0006003
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005996, upper bound: 0.0006043
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006228, upper bound: 0.0005960
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005951, upper bound: 0.0006205
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006052, upper bound: 0.0006007
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006043
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006145, upper bound: 0.0005939
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005931, upper bound: 0.0006111
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006061, upper bound: 0.0006008
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005995, upper bound: 0.0006037
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006228, upper bound: 0.0005961
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005950, upper bound: 0.0006206
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0005994, upper bound: 0.0006036
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006062, upper bound: 0.0006012
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.13
Output dim: 2, lower bound: -0.0006147, upper bound: 0.0005939

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0042018, -0.0041314, -0.0042018, -0.0041314, -0.0000704, 0.0000704
1: -0.0100144, -0.0092354, -0.0100144, -0.0092354, -0.0007790, 0.0007790
2: 0.9644458, 0.9653805, 0.9644458, 0.9653805, -0.0009347, 0.0009347
3: -0.0159361, -0.0090411, -0.0159361, -0.0090411, -0.0049373, 0.0049130
4: -0.0000054, 0.0005190, -0.0000054, 0.0005190, -0.0005244, 0.0005244
5: 0.0172649, 0.0180460, 0.0172649, 0.0180460, -0.0007811, 0.0007811
6: 0.0025825, 0.0035092, 0.0025825, 0.0035092, -0.0009266, 0.0009266
7: -0.0054352, -0.0032511, -0.0054352, -0.0032511, -0.0021841, 0.0021841
8: 0.0124171, 0.0138348, 0.0124171, 0.0138348, -0.0014176, 0.0014176
9: 0.0200581, 0.0226078, 0.0200581, 0.0226078, -0.0023301, 0.0023327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.78 + 597.99 = 600.78 seconds
