## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0029930225


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872)
1: (0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107)
2: (-0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0057238, 0.0057238)
3: (0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783)
4: (0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503)
5: (0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701)
6: (-0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600)
7: (-0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067)
8: (-0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109517, 0.0109517)
9: (-0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.11 + 2.95 = 4.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0032357, upper bound: 0.0032357

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031900, upper bound: 0.0032156
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031900, upper bound: 0.0031900
time: 2.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.03 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.03
Output dim: 1, lower bound: -0.0031900, upper bound: 0.0032156
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.03
Output dim: 1, lower bound: -0.0031900, upper bound: 0.0031900

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872
1: 0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107
2: -0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0057063, 0.0057099
3: 0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783
4: 0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503
5: 0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701
6: -0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600
7: -0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067
8: -0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109480, 0.0109471
9: -0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031437, upper bound: 0.0031690
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031437, upper bound: 0.0031662
time: 2.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872
1: 0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107
2: -0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0057099, 0.0057063
3: 0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783
4: 0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503
5: 0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701
6: -0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600
7: -0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067
8: -0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109471, 0.0109480
9: -0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031437, upper bound: 0.0031445
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031437, upper bound: 0.0031436
time: 2.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.88
Output dim: 1, lower bound: -0.0031437, upper bound: 0.0031690
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.88
Output dim: 1, lower bound: -0.0031437, upper bound: 0.0031662
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.88
Output dim: 1, lower bound: -0.0031437, upper bound: 0.0031445
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.88
Output dim: 1, lower bound: -0.0031437, upper bound: 0.0031436

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872
1: 0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107
2: -0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0056861, 0.0056947
3: 0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783
4: 0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503
5: 0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701
6: -0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600
7: -0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067
8: -0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109444, 0.0109421
9: -0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025728, upper bound: 0.0025753
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025728, upper bound: 0.0025753
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872
1: 0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107
2: -0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0056912, 0.0056898
3: 0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783
4: 0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503
5: 0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701
6: -0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600
7: -0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067
8: -0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109431, 0.0109434
9: -0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025733, upper bound: 0.0025742
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025733, upper bound: 0.0025742
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872
1: 0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107
2: -0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0056898, 0.0056912
3: 0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783
4: 0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503
5: 0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701
6: -0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600
7: -0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067
8: -0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109434, 0.0109431
9: -0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025745, upper bound: 0.0025731
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025745, upper bound: 0.0025731
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872
1: 0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107
2: -0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0056947, 0.0056861
3: 0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783
4: 0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503
5: 0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701
6: -0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600
7: -0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067
8: -0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109421, 0.0109444
9: -0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025753, upper bound: 0.0025728
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0025753, upper bound: 0.0025728
time: 1.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 1, lower bound: -0.0025728, upper bound: 0.0025753
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 1, lower bound: -0.0025728, upper bound: 0.0025753
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 1, lower bound: -0.0025733, upper bound: 0.0025742
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 1, lower bound: -0.0025733, upper bound: 0.0025742
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 1, lower bound: -0.0025745, upper bound: 0.0025731
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 1, lower bound: -0.0025745, upper bound: 0.0025731
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 1, lower bound: -0.0025753, upper bound: 0.0025728
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 1, lower bound: -0.0025753, upper bound: 0.0025728

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.06 + 30.62 = 34.68 seconds
