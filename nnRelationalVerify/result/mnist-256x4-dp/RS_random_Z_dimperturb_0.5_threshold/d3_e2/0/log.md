## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001618947


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0032526, 0.0032526)
1: (-0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011393, 0.0011393)
2: (0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0043218, 0.0043218)
3: (1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396)
4: (-0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0007033, 0.0007033)
5: (0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0025027, 0.0025027)
6: (-0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924)
7: (-0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046424, 0.0046424)
8: (-0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075999, 0.0075999)
9: (-0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0037531, 0.0037531)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.44 + 1.96 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0019582, upper bound: 0.0019582

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019486, upper bound: 0.0019515
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019515, upper bound: 0.0019486
time: 1.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.03 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.03
Output dim: 3, lower bound: -0.0019486, upper bound: 0.0019515
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.03
Output dim: 3, lower bound: -0.0019515, upper bound: 0.0019486

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0032404, 0.0032351
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011337, 0.0011311
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0043031, 0.0042949
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006982, 0.0006998
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024931, 0.0024889
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046403, 0.0046409
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075415, 0.0075592
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0037314, 0.0037219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019040, upper bound: 0.0019054
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019040, upper bound: 0.0019054
time: 1.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0032351, 0.0032526
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011311, 0.0011393
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042949, 0.0043218
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0007033, 0.0006982
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024889, 0.0025027
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046424, 0.0046403
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075999, 0.0075415
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0037219, 0.0037531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017757, upper bound: 0.0017757
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017757, upper bound: 0.0017757
time: 1.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -0.0019040, upper bound: 0.0019054
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -0.0019040, upper bound: 0.0019054
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -0.0017757, upper bound: 0.0017757
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -0.0017757, upper bound: 0.0017757

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031968, 0.0031973
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011144, 0.0011147
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042361, 0.0042368
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006874, 0.0006873
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024588, 0.0024592
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046359, 0.0046358
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074156, 0.0074140
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036538, 0.0036546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018590, upper bound: 0.0018676
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018661, upper bound: 0.0018594
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0032026, 0.0032351
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011172, 0.0011311
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042450, 0.0042949
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006982, 0.0006889
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024634, 0.0024889
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046403, 0.0046365
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075415, 0.0074333
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036641, 0.0037219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017516, upper bound: 0.0017524
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017516, upper bound: 0.0017524
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0032283, 0.0032498
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011277, 0.0011378
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042845, 0.0043175
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0007025, 0.0006963
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024836, 0.0025005
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046420, 0.0046395
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075905, 0.0075190
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0037099, 0.0037481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016488, upper bound: 0.0016488
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016488, upper bound: 0.0016488
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0032323, 0.0032526
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011296, 0.0011393
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042906, 0.0043218
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0007033, 0.0006974
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024868, 0.0025027
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046424, 0.0046400
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075999, 0.0075323
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0037170, 0.0037531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016488, upper bound: 0.0016488
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016488, upper bound: 0.0016488
time: 0.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 3, lower bound: -0.0018590, upper bound: 0.0018676
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 3, lower bound: -0.0018661, upper bound: 0.0018594
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 3, lower bound: -0.0017516, upper bound: 0.0017524
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 3, lower bound: -0.0017516, upper bound: 0.0017524
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 3, lower bound: -0.0016488, upper bound: 0.0016488
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 3, lower bound: -0.0016488, upper bound: 0.0016488
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 3, lower bound: -0.0016488, upper bound: 0.0016488
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 3, lower bound: -0.0016488, upper bound: 0.0016488

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031058, 0.0030986
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010670, 0.0010635
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040882, 0.0040771
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006572, 0.0006592
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023863, 0.0023807
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046268, 0.0046276
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070594, 0.0070836
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034732, 0.0034603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017315, upper bound: 0.0017384
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017315, upper bound: 0.0017385
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030981, 0.0031047
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010632, 0.0010664
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040764, 0.0040864
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006589, 0.0006570
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023803, 0.0023854
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046275, 0.0046267
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070797, 0.0070579
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034595, 0.0034712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017384, upper bound: 0.0017316
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017384, upper bound: 0.0017316
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031972, 0.0032375
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011147, 0.0011324
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042366, 0.0042986
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006989, 0.0006874
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024591, 0.0024908
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046406, 0.0046359
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075499, 0.0074156
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036547, 0.0037264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017277, upper bound: 0.0017304
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017299, upper bound: 0.0017282
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0032026, 0.0032297
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011172, 0.0011285
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042450, 0.0042865
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006967, 0.0006889
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024634, 0.0024847
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046397, 0.0046365
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075238, 0.0074333
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036641, 0.0037125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017218, upper bound: 0.0017227
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017218, upper bound: 0.0017227
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031852, 0.0032121
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011086, 0.0011216
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042182, 0.0042595
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006917, 0.0006840
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024497, 0.0024708
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046376, 0.0046345
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074649, 0.0073754
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036331, 0.0036810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016212, upper bound: 0.0016043
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016043, upper bound: 0.0016212
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031905, 0.0032498
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011112, 0.0011378
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042264, 0.0043175
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0007025, 0.0006855
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024539, 0.0025005
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046420, 0.0046351
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075905, 0.0073930
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036426, 0.0037481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016212, upper bound: 0.0016043
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016043, upper bound: 0.0016212
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031893, 0.0032150
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011106, 0.0011231
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042246, 0.0042639
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006925, 0.0006851
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024529, 0.0024731
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046380, 0.0046350
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074744, 0.0073891
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036405, 0.0036861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015987, upper bound: 0.0015987
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015987, upper bound: 0.0015987
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031945, 0.0032526
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011131, 0.0011393
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042325, 0.0043218
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0007033, 0.0006866
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024570, 0.0025027
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046424, 0.0046356
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0075999, 0.0074063
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036497, 0.0037531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014202, upper bound: 0.0014202
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0014202, upper bound: 0.0014202
time: 0.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0017315, upper bound: 0.0017384
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0017315, upper bound: 0.0017385
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0017384, upper bound: 0.0017316
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0017384, upper bound: 0.0017316
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0017277, upper bound: 0.0017304
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0017299, upper bound: 0.0017282
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0017218, upper bound: 0.0017227
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0017218, upper bound: 0.0017227
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0016212, upper bound: 0.0016043
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0016043, upper bound: 0.0016212
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0016212, upper bound: 0.0016043
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0016043, upper bound: 0.0016212
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0015987, upper bound: 0.0015987
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0015987, upper bound: 0.0015987
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0014202, upper bound: 0.0014202
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 3, lower bound: -0.0014202, upper bound: 0.0014202

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030995, 0.0030969
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010639, 0.0010626
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040781, 0.0040743
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006567, 0.0006574
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023813, 0.0023793
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046266, 0.0046269
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070536, 0.0070620
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034618, 0.0034573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0016434
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0016434
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031058, 0.0030922
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010670, 0.0010603
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040882, 0.0040670
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006553, 0.0006592
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023863, 0.0023756
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046260, 0.0046276
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070379, 0.0070836
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034732, 0.0034489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017243, upper bound: 0.0017310
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017243, upper bound: 0.0017310
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030917, 0.0031014
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010601, 0.0010648
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040663, 0.0040812
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006580, 0.0006552
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023752, 0.0023828
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046271, 0.0046260
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070686, 0.0070363
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034481, 0.0034653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017310, upper bound: 0.0017243
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017310, upper bound: 0.0017243
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030981, 0.0030983
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010632, 0.0010633
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040764, 0.0040763
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006571, 0.0006570
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023803, 0.0023804
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046268, 0.0046267
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070581, 0.0070579
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034595, 0.0034597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016434, upper bound: 0.0016400
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016434, upper bound: 0.0016400
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031785, 0.0032181
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011050, 0.0011223
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042067, 0.0042676
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006932, 0.0006818
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024443, 0.0024755
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046382, 0.0046335
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074820, 0.0073500
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036194, 0.0036899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016891, upper bound: 0.0016904
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016891, upper bound: 0.0016904
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031783, 0.0032188
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011049, 0.0011227
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0042065, 0.0042687
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006934, 0.0006818
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024441, 0.0024760
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046383, 0.0046335
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074843, 0.0073495
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036191, 0.0036911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016980, upper bound: 0.0016679
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016699, upper bound: 0.0016961
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031696, 0.0031963
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011007, 0.0011117
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041940, 0.0042350
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006873, 0.0006796
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024373, 0.0024584
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046356, 0.0046325
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074141, 0.0073248
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036040, 0.0036515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016743, upper bound: 0.0016756
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016743, upper bound: 0.0016756
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031689, 0.0031966
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0011004, 0.0011119
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041929, 0.0042354
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006873, 0.0006794
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024368, 0.0024586
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046357, 0.0046324
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074150, 0.0073225
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0036028, 0.0036520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016743, upper bound: 0.0016756
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016743, upper bound: 0.0016756
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030828, 0.0031495
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010670, 0.0010992
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040579, 0.0041605
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006739, 0.0006548
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023686, 0.0024212
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046294, 0.0046216
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0072716, 0.0070493
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034650, 0.0035838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016002, upper bound: 0.0015822
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016002, upper bound: 0.0015822
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031190, 0.0031076
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010847, 0.0010787
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041136, 0.0040961
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006619, 0.0006652
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023971, 0.0023882
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046245, 0.0046258
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0071319, 0.0071699
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035295, 0.0035092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0013882, upper bound: 0.0013920
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0013882, upper bound: 0.0013920
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030881, 0.0031902
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010696, 0.0011173
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040661, 0.0042231
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006856, 0.0006564
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023728, 0.0024532
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046342, 0.0046222
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074073, 0.0070669
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034744, 0.0036563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015697, upper bound: 0.0015520
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015697, upper bound: 0.0015520
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031302, 0.0031483
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010902, 0.0010969
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041309, 0.0041587
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006736, 0.0006684
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024060, 0.0024202
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046292, 0.0046271
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0072676, 0.0072073
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035495, 0.0035817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015520, upper bound: 0.0015697
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015520, upper bound: 0.0015697
time: 0.88 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0016434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016400, upper bound: 0.0016434
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0017243, upper bound: 0.0017310
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0017243, upper bound: 0.0017310
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0017310, upper bound: 0.0017243
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0017310, upper bound: 0.0017243
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016434, upper bound: 0.0016400
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016434, upper bound: 0.0016400
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016891, upper bound: 0.0016904
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016891, upper bound: 0.0016904
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016980, upper bound: 0.0016679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016699, upper bound: 0.0016961
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016743, upper bound: 0.0016756
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016743, upper bound: 0.0016756
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016743, upper bound: 0.0016756
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016743, upper bound: 0.0016756
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016002, upper bound: 0.0015822
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0016002, upper bound: 0.0015822
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0013882, upper bound: 0.0013920
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0013882, upper bound: 0.0013920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0015697, upper bound: 0.0015520
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0015697, upper bound: 0.0015520
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0015520, upper bound: 0.0015697
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.09
Output dim: 3, lower bound: -0.0015520, upper bound: 0.0015697

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030644, 0.0030636
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010469, 0.0010465
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040242, 0.0040230
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006471, 0.0006474
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023537, 0.0023530
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046227, 0.0046228
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069424, 0.0069452
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033994, 0.0033979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016085, upper bound: 0.0016099
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016072, upper bound: 0.0016110
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030661, 0.0030969
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010477, 0.0010626
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040268, 0.0040743
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006567, 0.0006478
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023550, 0.0023793
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046266, 0.0046230
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070536, 0.0069508
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034024, 0.0034573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016143, upper bound: 0.0015904
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015833, upper bound: 0.0016182
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030723, 0.0030584
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010505, 0.0010438
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040364, 0.0040147
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006457, 0.0006497
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023599, 0.0023490
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046220, 0.0046236
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069272, 0.0069737
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034136, 0.0033891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016316, upper bound: 0.0016351
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016316, upper bound: 0.0016351
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030710, 0.0030587
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010499, 0.0010440
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040344, 0.0040151
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006458, 0.0006493
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023589, 0.0023492
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046220, 0.0046235
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069282, 0.0069694
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034113, 0.0033896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016908, upper bound: 0.0016956
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016894, upper bound: 0.0016972
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030583, 0.0030676
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010437, 0.0010483
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040144, 0.0040287
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006483, 0.0006457
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023489, 0.0023562
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046231, 0.0046220
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069576, 0.0069266
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033888, 0.0034053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017081, upper bound: 0.0017038
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0017098, upper bound: 0.0017028
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030576, 0.0030679
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010434, 0.0010485
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040134, 0.0040293
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006484, 0.0006455
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023483, 0.0023565
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046231, 0.0046219
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069588, 0.0069243
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033876, 0.0034060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016351, upper bound: 0.0016316
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016351, upper bound: 0.0016316
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030615, 0.0030649
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010455, 0.0010471
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040201, 0.0040250
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006475, 0.0006465
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023515, 0.0023541
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046228, 0.0046224
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069469, 0.0069359
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033943, 0.0034003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016255, upper bound: 0.0016262
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016282, upper bound: 0.0016242
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030647, 0.0030983
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010470, 0.0010633
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040251, 0.0040763
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006571, 0.0006475
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023540, 0.0023804
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046268, 0.0046228
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070581, 0.0069467
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034001, 0.0034597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016352, upper bound: 0.0016316
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016351, upper bound: 0.0016316
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031449, 0.0031833
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010884, 0.0011051
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041551, 0.0042142
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006834, 0.0006724
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024179, 0.0024481
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046340, 0.0046295
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0073686, 0.0072406
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035591, 0.0036274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016468, upper bound: 0.0016507
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016482, upper bound: 0.0016477
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031442, 0.0031845
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010880, 0.0011057
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041541, 0.0042161
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006837, 0.0006722
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024174, 0.0024491
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046341, 0.0046294
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0073727, 0.0072384
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035579, 0.0036296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016377, upper bound: 0.0016397
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016377, upper bound: 0.0016397
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030754, 0.0031612
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010637, 0.0011042
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040464, 0.0041782
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006774, 0.0006529
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023629, 0.0024304
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046306, 0.0046206
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0073112, 0.0070255
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034525, 0.0036052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016431, upper bound: 0.0016142
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016431, upper bound: 0.0016142
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031199, 0.0031193
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010854, 0.0010837
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041147, 0.0041138
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006654, 0.0006656
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023978, 0.0023974
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046257, 0.0046258
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0071717, 0.0071736
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035316, 0.0035306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016155, upper bound: 0.0016415
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016155, upper bound: 0.0016415
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031351, 0.0031630
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010833, 0.0010944
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041411, 0.0041838
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006777, 0.0006697
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024102, 0.0024322
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046317, 0.0046285
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0073033, 0.0072101
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035427, 0.0035923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016335, upper bound: 0.0016373
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016363, upper bound: 0.0016347
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031368, 0.0031963
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010841, 0.0011117
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041435, 0.0042350
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006873, 0.0006702
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024115, 0.0024584
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046356, 0.0046287
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074141, 0.0072155
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035456, 0.0036515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016335, upper bound: 0.0016373
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016363, upper bound: 0.0016347
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031366, 0.0031633
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010840, 0.0010946
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041433, 0.0041843
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006778, 0.0006701
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024114, 0.0024324
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046318, 0.0046286
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0073042, 0.0072149
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035453, 0.0035928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016431, upper bound: 0.0016170
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016170, upper bound: 0.0016444
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031361, 0.0031966
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010837, 0.0011119
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041425, 0.0042354
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006873, 0.0006700
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024110, 0.0024586
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046357, 0.0046286
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0074150, 0.0072133
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035444, 0.0036520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016339, upper bound: 0.0016349
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016339, upper bound: 0.0016349
time: 1.17 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016085, upper bound: 0.0016099
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016072, upper bound: 0.0016110
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016143, upper bound: 0.0015904
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0015833, upper bound: 0.0016182
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016316, upper bound: 0.0016351
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016316, upper bound: 0.0016351
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016908, upper bound: 0.0016956
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016894, upper bound: 0.0016972
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0017081, upper bound: 0.0017038
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0017098, upper bound: 0.0017028
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016351, upper bound: 0.0016316
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016351, upper bound: 0.0016316
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016255, upper bound: 0.0016262
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016282, upper bound: 0.0016242
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016352, upper bound: 0.0016316
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016351, upper bound: 0.0016316
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016468, upper bound: 0.0016507
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016482, upper bound: 0.0016477
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016377, upper bound: 0.0016397
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016377, upper bound: 0.0016397
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016431, upper bound: 0.0016142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016431, upper bound: 0.0016142
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016155, upper bound: 0.0016415
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016155, upper bound: 0.0016415
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016335, upper bound: 0.0016373
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016363, upper bound: 0.0016347
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016335, upper bound: 0.0016373
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016363, upper bound: 0.0016347
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016431, upper bound: 0.0016170
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016170, upper bound: 0.0016444
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016339, upper bound: 0.0016349
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.95
Output dim: 3, lower bound: -0.0016339, upper bound: 0.0016349

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030372, 0.0030251
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010335, 0.0010276
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039825, 0.0039634
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006362, 0.0006397
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023323, 0.0023227
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046181, 0.0046195
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0068160, 0.0068569
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033511, 0.0033297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015984, upper bound: 0.0015991
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015968, upper bound: 0.0016002
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030389, 0.0030584
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010343, 0.0010438
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039851, 0.0040147
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006457, 0.0006401
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023337, 0.0023490
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046220, 0.0046197
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069272, 0.0068625
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033542, 0.0033891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016059, upper bound: 0.0015799
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015731, upper bound: 0.0016092
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030375, 0.0030267
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010318, 0.0010265
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039787, 0.0039618
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006356, 0.0006387
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023324, 0.0023239
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046185, 0.0046198
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0068092, 0.0068454
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033455, 0.0033260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016704, upper bound: 0.0016739
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016701, upper bound: 0.0016748
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030710, 0.0030255
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010499, 0.0010259
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040344, 0.0039600
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006352, 0.0006493
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023589, 0.0023230
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046183, 0.0046235
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0068052, 0.0069694
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034113, 0.0033239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016610, upper bound: 0.0016698
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016617, upper bound: 0.0016688
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030379, 0.0030465
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010333, 0.0010376
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039832, 0.0039965
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006420, 0.0006396
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023328, 0.0023396
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046206, 0.0046196
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0068851, 0.0068562
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033509, 0.0033663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016688, upper bound: 0.0016617
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016668, upper bound: 0.0016640
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030377, 0.0030472
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010332, 0.0010379
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039829, 0.0039975
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006422, 0.0006395
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023327, 0.0023401
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046207, 0.0046196
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0068872, 0.0068556
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033505, 0.0033675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016814, upper bound: 0.0016392
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016462, upper bound: 0.0016747
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030226, 0.0030346
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010264, 0.0010323
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039595, 0.0039780
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006389, 0.0006354
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023208, 0.0023302
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046192, 0.0046178
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0068477, 0.0068077
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033252, 0.0033466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016146, upper bound: 0.0016120
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016139, upper bound: 0.0016131
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030242, 0.0030679
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010272, 0.0010485
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039621, 0.0040293
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006484, 0.0006359
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023221, 0.0023565
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046231, 0.0046180
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069588, 0.0068131
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033282, 0.0034060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016092, upper bound: 0.0015731
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015799, upper bound: 0.0016059
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030417, 0.0030444
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010351, 0.0010363
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039894, 0.0039934
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006415, 0.0006407
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023358, 0.0023380
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046204, 0.0046201
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0068759, 0.0068672
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033565, 0.0033611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015970, upper bound: 0.0015665
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015701, upper bound: 0.0015985
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030410, 0.0030451
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010347, 0.0010367
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039885, 0.0039944
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006417, 0.0006405
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023353, 0.0023385
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046205, 0.0046200
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0068782, 0.0068650
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033554, 0.0033623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016118, upper bound: 0.0016095
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016116, upper bound: 0.0016095
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030312, 0.0030630
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010306, 0.0010460
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039732, 0.0040217
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006470, 0.0006379
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023276, 0.0023526
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046225, 0.0046188
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069423, 0.0068368
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033404, 0.0033972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016092, upper bound: 0.0015733
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015799, upper bound: 0.0016059
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030305, 0.0030648
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010302, 0.0010469
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039722, 0.0040245
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006475, 0.0006377
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023271, 0.0023540
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046227, 0.0046187
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069484, 0.0068345
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033392, 0.0034004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016107, upper bound: 0.0016105
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016116, upper bound: 0.0016095
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030497, 0.0030831
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010389, 0.0010532
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040012, 0.0040525
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006525, 0.0006429
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023421, 0.0023684
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046248, 0.0046209
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070062, 0.0068949
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033710, 0.0034305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016133, upper bound: 0.0015895
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015837, upper bound: 0.0016172
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030449, 0.0030886
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010366, 0.0010559
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039939, 0.0040610
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006541, 0.0006416
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023384, 0.0023728
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046255, 0.0046204
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070246, 0.0068791
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033626, 0.0034403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015900, upper bound: 0.0015876
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015900, upper bound: 0.0015876
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031119, 0.0031513
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010716, 0.0010884
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041044, 0.0041649
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006742, 0.0006629
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023920, 0.0024229
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046302, 0.0046256
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0072618, 0.0071308
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035004, 0.0035704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0015678
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0015678
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031115, 0.0031845
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010714, 0.0011057
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041037, 0.0042161
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006837, 0.0006628
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023916, 0.0024491
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046341, 0.0046256
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0073727, 0.0071291
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034995, 0.0036296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0015678
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0015678
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030709, 0.0031563
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010613, 0.0011016
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040394, 0.0041706
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006760, 0.0006516
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023593, 0.0024265
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046301, 0.0046201
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0072940, 0.0070097
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034440, 0.0035958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016021, upper bound: 0.0015740
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016021, upper bound: 0.0015740
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030754, 0.0031567
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010637, 0.0011018
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040464, 0.0041712
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006761, 0.0006529
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023629, 0.0024268
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046301, 0.0046206
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0072954, 0.0070255
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034525, 0.0035966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015784, upper bound: 0.0015524
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015784, upper bound: 0.0015524
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031154, 0.0031146
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010830, 0.0010812
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041077, 0.0041065
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006641, 0.0006643
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023943, 0.0023937
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046252, 0.0046253
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0071552, 0.0071578
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035231, 0.0035217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015765, upper bound: 0.0016007
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015765, upper bound: 0.0016007
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031199, 0.0031148
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010854, 0.0010813
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041147, 0.0041068
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006641, 0.0006656
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023978, 0.0023938
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046252, 0.0046258
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0071559, 0.0071736
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035316, 0.0035220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015745, upper bound: 0.0016014
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015745, upper bound: 0.0016014
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030420, 0.0030634
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010358, 0.0010439
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039898, 0.0040224
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006471, 0.0006410
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023361, 0.0023530
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046225, 0.0046201
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069435, 0.0068727
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033596, 0.0033969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015972, upper bound: 0.0016015
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015981, upper bound: 0.0015999
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030354, 0.0030680
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010326, 0.0010461
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039796, 0.0040293
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006484, 0.0006391
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023309, 0.0023565
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046230, 0.0046193
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069585, 0.0068507
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033478, 0.0034050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015986, upper bound: 0.0015990
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015983
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030417, 0.0030972
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010357, 0.0010604
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039894, 0.0040743
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006568, 0.0006409
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023359, 0.0023795
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046265, 0.0046200
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070559, 0.0068719
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033592, 0.0034570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015972, upper bound: 0.0016015
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015981, upper bound: 0.0015999
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030370, 0.0031017
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010334, 0.0010627
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039821, 0.0040812
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006581, 0.0006396
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023321, 0.0023831
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046270, 0.0046195
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0070710, 0.0068561
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033507, 0.0034651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015986, upper bound: 0.0015990
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015983
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030337, 0.0031062
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010438, 0.0010778
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039833, 0.0040945
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006616, 0.0006409
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023301, 0.0023871
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046241, 0.0046156
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0071265, 0.0068854
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033777, 0.0035065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015757
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015757
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030821, 0.0030643
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010675, 0.0010573
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040578, 0.0040301
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006496, 0.0006547
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023682, 0.0023541
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046192, 0.0046213
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0069868, 0.0070468
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0034639, 0.0034319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015741, upper bound: 0.0016058
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015791, upper bound: 0.0016019
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031316, 0.0031921
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010814, 0.0011095
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041354, 0.0042283
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006860, 0.0006687
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024074, 0.0024550
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046351, 0.0046280
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0073995, 0.0071979
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035358, 0.0036434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015757
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015757, upper bound: 0.0016015
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0031361, 0.0031921
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010837, 0.0011095
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0041425, 0.0042283
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006860, 0.0006700
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0024110, 0.0024550
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046351, 0.0046286
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0073996, 0.0072133
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0035444, 0.0036435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015854, upper bound: 0.0015867
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015857, upper bound: 0.0015866
time: 1.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015984, upper bound: 0.0015991
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015968, upper bound: 0.0016002
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016059, upper bound: 0.0015799
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015731, upper bound: 0.0016092
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016704, upper bound: 0.0016739
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016701, upper bound: 0.0016748
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016610, upper bound: 0.0016698
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016617, upper bound: 0.0016688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016688, upper bound: 0.0016617
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016668, upper bound: 0.0016640
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016814, upper bound: 0.0016392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016462, upper bound: 0.0016747
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016146, upper bound: 0.0016120
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016139, upper bound: 0.0016131
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016092, upper bound: 0.0015731
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015799, upper bound: 0.0016059
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015970, upper bound: 0.0015665
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015701, upper bound: 0.0015985
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016118, upper bound: 0.0016095
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016116, upper bound: 0.0016095
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016092, upper bound: 0.0015733
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015799, upper bound: 0.0016059
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016107, upper bound: 0.0016105
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016116, upper bound: 0.0016095
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016133, upper bound: 0.0015895
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015837, upper bound: 0.0016172
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015900, upper bound: 0.0015876
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015900, upper bound: 0.0015876
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0015678
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0015678
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0015678
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0015678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016021, upper bound: 0.0015740
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016021, upper bound: 0.0015740
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015784, upper bound: 0.0015524
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015784, upper bound: 0.0015524
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015765, upper bound: 0.0016007
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015765, upper bound: 0.0016007
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015745, upper bound: 0.0016014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015745, upper bound: 0.0016014
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015972, upper bound: 0.0016015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015981, upper bound: 0.0015999
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015986, upper bound: 0.0015990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015983
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015972, upper bound: 0.0016015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015981, upper bound: 0.0015999
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015986, upper bound: 0.0015990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015983
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015741, upper bound: 0.0016058
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015791, upper bound: 0.0016019
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0016007, upper bound: 0.0015757
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015757, upper bound: 0.0016015
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015854, upper bound: 0.0015867
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 3, lower bound: -0.0015857, upper bound: 0.0015866

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030328, 0.0030219
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010295, 0.0010241
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039715, 0.0039544
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006342, 0.0006374
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023288, 0.0023201
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046179, 0.0046192
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067928, 0.0068294
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033370, 0.0033173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015785, upper bound: 0.0015781
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015785, upper bound: 0.0015781
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030375, 0.0030220
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010318, 0.0010242
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039787, 0.0039546
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006342, 0.0006387
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023324, 0.0023202
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046179, 0.0046198
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067932, 0.0068454
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033455, 0.0033175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016328, upper bound: 0.0016388
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016330, upper bound: 0.0016375
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030506, 0.0030047
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010396, 0.0010151
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040031, 0.0039279
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006291, 0.0006433
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023429, 0.0023066
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046159, 0.0046211
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067331, 0.0068988
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033736, 0.0032845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016325, upper bound: 0.0016038
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015975, upper bound: 0.0016415
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030497, 0.0030051
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010391, 0.0010153
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0040017, 0.0039286
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006292, 0.0006430
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023422, 0.0023070
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046160, 0.0046210
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067345, 0.0068958
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033720, 0.0032852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016330, upper bound: 0.0016033
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0015977, upper bound: 0.0016402
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030047, 0.0030176
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010150, 0.0010214
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039279, 0.0039477
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006328, 0.0006291
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023066, 0.0023168
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046174, 0.0046159
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067760, 0.0067329
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0032844, 0.0033074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016402, upper bound: 0.0015977
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016033, upper bound: 0.0016330
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030379, 0.0030133
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010333, 0.0010193
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039832, 0.0039412
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006316, 0.0006396
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023328, 0.0023134
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046169, 0.0046196
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067618, 0.0068562
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033509, 0.0032998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016375, upper bound: 0.0016330
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016354, upper bound: 0.0016345
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029265, 0.0029788
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0009924, 0.0010180
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038136, 0.0038940
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031394
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006239, 0.0006089
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022453, 0.0022865
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046121, 0.0046059
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0066811, 0.0065067
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0031720, 0.0032651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016415, upper bound: 0.0015976
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016392, upper bound: 0.0015976
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029644, 0.0029360
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010110, 0.0009971
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038719, 0.0038282
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006116, 0.0006198
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022751, 0.0022528
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046070, 0.0046104
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0065384, 0.0066331
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0032395, 0.0031889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016038, upper bound: 0.0016325
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0016038, upper bound: 0.0016351
time: 1.30 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 11.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0015785, upper bound: 0.0015781
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0015785, upper bound: 0.0015781
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016328, upper bound: 0.0016388
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016330, upper bound: 0.0016375
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016325, upper bound: 0.0016038
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0015975, upper bound: 0.0016415
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016330, upper bound: 0.0016033
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0015977, upper bound: 0.0016402
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016402, upper bound: 0.0015977
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016033, upper bound: 0.0016330
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016375, upper bound: 0.0016330
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016354, upper bound: 0.0016345
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016415, upper bound: 0.0015976
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016392, upper bound: 0.0015976
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016038, upper bound: 0.0016325
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 11.80
Output dim: 3, lower bound: -0.0016038, upper bound: 0.0016351

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030171, 0.0030019
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010212, 0.0010137
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039473, 0.0039233
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006282, 0.0006327
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023164, 0.0023043
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046156, 0.0046174
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067231, 0.0067745
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033066, 0.0032793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015390, upper bound: 0.0015424
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015390, upper bound: 0.0015424
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030162, 0.0030018
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010207, 0.0010136
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039459, 0.0039231
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006282, 0.0006325
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023157, 0.0023043
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046156, 0.0046173
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067227, 0.0067714
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033050, 0.0032792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015391, upper bound: 0.0015416
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015391, upper bound: 0.0015416
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029393, 0.0029361
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0009986, 0.0009956
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038336, 0.0038251
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006106, 0.0006126
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022554, 0.0022527
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046072, 0.0046074
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0065270, 0.0065493
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0031945, 0.0031834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015399, upper bound: 0.0015137
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015399, upper bound: 0.0015137
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029783, 0.0028938
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010177, 0.0009749
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038935, 0.0037601
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0030956, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0005985, 0.0006238
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022860, 0.0022193
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046023, 0.0046120
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0063860, 0.0066791
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0032639, 0.0031080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015078, upper bound: 0.0015439
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015078, upper bound: 0.0015439
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029384, 0.0029361
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0009982, 0.0009956
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038322, 0.0038252
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006106, 0.0006124
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022547, 0.0022527
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046072, 0.0046073
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0065271, 0.0065463
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0031929, 0.0031834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015410, upper bound: 0.0015129
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015410, upper bound: 0.0015129
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029768, 0.0028942
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010169, 0.0009751
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038912, 0.0037607
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0030961, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0005986, 0.0006233
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022849, 0.0022197
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046023, 0.0046118
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0063874, 0.0066741
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0032612, 0.0031088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0016082
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016105
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0028937, 0.0029509
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0009748, 0.0010028
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0037600, 0.0038478
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0030956
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006149, 0.0005985
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022193, 0.0022643
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046090, 0.0046023
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0065762, 0.0063859
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0031079, 0.0032097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016105, upper bound: 0.0015656
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016082, upper bound: 0.0015659
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029319, 0.0029066
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0009935, 0.0009812
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038186, 0.0037799
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031113, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006022, 0.0006094
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022493, 0.0022295
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046038, 0.0046067
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0064289, 0.0065129
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0031758, 0.0031309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015129, upper bound: 0.0015410
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015129, upper bound: 0.0015410
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030331, 0.0030084
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010310, 0.0010168
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039758, 0.0039333
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006301, 0.0006382
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023291, 0.0023095
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046163, 0.0046191
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067448, 0.0068405
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033426, 0.0032909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015416, upper bound: 0.0015391
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015416, upper bound: 0.0015391
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0030379, 0.0030088
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010333, 0.0010170
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0039832, 0.0039339
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006302, 0.0006396
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0023328, 0.0023098
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046164, 0.0046196
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0067460, 0.0068562
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0033509, 0.0032916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0016058, upper bound: 0.0015660
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015715, upper bound: 0.0016056
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0028935, 0.0029517
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0009747, 0.0010032
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0037597, 0.0038492
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0030953
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006151, 0.0005984
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022192, 0.0022650
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046091, 0.0046022
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0065792, 0.0063852
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0031076, 0.0032112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015439, upper bound: 0.0015078
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015439, upper bound: 0.0015078
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029265, 0.0029458
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0009924, 0.0010003
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038136, 0.0038401
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031396, 0.0031394
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006134, 0.0006089
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022453, 0.0022603
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046084, 0.0046059
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0065596, 0.0065067
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0031720, 0.0032008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015425, upper bound: 0.0015087
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015425, upper bound: 0.0015087
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029315, 0.0029076
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0009933, 0.0009816
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038180, 0.0037814
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031125, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006025, 0.0006093
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022490, 0.0022303
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046039, 0.0046067
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0064322, 0.0065116
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0031751, 0.0031327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015137, upper bound: 0.0015399
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015137, upper bound: 0.0015399
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0028670, 0.0008039, -0.0028670, 0.0008039, -0.0029644, 0.0029030
1: -0.0045949, -0.0033714, -0.0045949, -0.0033714, -0.0010110, 0.0009794
2: 0.0109632, 0.0159135, 0.0109632, 0.0159135, -0.0038719, 0.0037743
3: 1.0068085, 1.0099481, 1.0068085, 1.0099481, -0.0031069, 0.0031396
4: -0.0042305, -0.0034137, -0.0042305, -0.0034137, -0.0006012, 0.0006198
5: 0.0017600, 0.0045908, 0.0017600, 0.0045908, -0.0022751, 0.0022266
6: -0.0025985, -0.0023061, -0.0025985, -0.0023061, -0.0002924, 0.0002924
7: -0.0130901, -0.0083996, -0.0130901, -0.0083996, -0.0046033, 0.0046104
8: -0.0134529, -0.0045562, -0.0134529, -0.0045562, -0.0064169, 0.0066331
9: -0.0019502, 0.0024852, -0.0019502, 0.0024852, -0.0032395, 0.0031245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015725, upper bound: 0.0016038
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0015721, upper bound: 0.0016052
time: 1.41 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 6.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015390, upper bound: 0.0015424
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015390, upper bound: 0.0015424
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015391, upper bound: 0.0015416
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015391, upper bound: 0.0015416
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015399, upper bound: 0.0015137
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015399, upper bound: 0.0015137
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015078, upper bound: 0.0015439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015078, upper bound: 0.0015439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015410, upper bound: 0.0015129
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015410, upper bound: 0.0015129
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015659, upper bound: 0.0016082
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015656, upper bound: 0.0016105
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0016105, upper bound: 0.0015656
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0016082, upper bound: 0.0015659
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015129, upper bound: 0.0015410
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015129, upper bound: 0.0015410
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015416, upper bound: 0.0015391
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015416, upper bound: 0.0015391
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0016058, upper bound: 0.0015660
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015715, upper bound: 0.0016056
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015439, upper bound: 0.0015078
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015439, upper bound: 0.0015078
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015425, upper bound: 0.0015087
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015425, upper bound: 0.0015087
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015137, upper bound: 0.0015399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015137, upper bound: 0.0015399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015725, upper bound: 0.0016038
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 6.31
Output dim: 3, lower bound: -0.0015721, upper bound: 0.0016052

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.40 + 482.72 = 486.12 seconds
