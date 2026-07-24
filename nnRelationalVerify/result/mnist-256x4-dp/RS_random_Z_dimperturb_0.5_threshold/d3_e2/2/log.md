## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00076797


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000288, 0.0000288)
1: (-0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0010793, 0.0010793)
2: (0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0012952, 0.0012952)
3: (0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0095530, 0.0095530)
4: (-0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0007266, 0.0007266)
5: (0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0007343, 0.0007343)
6: (0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003572, 0.0003572)
7: (-0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0024757, 0.0024757)
8: (0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0019641, 0.0019641)
9: (0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0035327, 0.0035327)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.58 = 3.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0008766, upper bound: 0.0008767

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008289, upper bound: 0.0008251
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008251, upper bound: 0.0008289
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 2, lower bound: -0.0008289, upper bound: 0.0008251
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 2, lower bound: -0.0008251, upper bound: 0.0008289

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000257, 0.0000257
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009624, 0.0009620
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011549, 0.0011544
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0085185, 0.0085146
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006476, 0.0006479
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006545, 0.0006548
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003185, 0.0003183
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0022066, 0.0022076
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0017506, 0.0017514
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0031487, 0.0031501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008159, upper bound: 0.0008123
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008160, upper bound: 0.0008121
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000257, 0.0000257
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009620, 0.0009624
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011544, 0.0011549
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0085145, 0.0085185
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006479, 0.0006476
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006548, 0.0006545
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003183, 0.0003185
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0022076, 0.0022066
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0017514, 0.0017506
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0031501, 0.0031487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007986, upper bound: 0.0008028
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007986, upper bound: 0.0008028
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 2, lower bound: -0.0008159, upper bound: 0.0008123
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 2, lower bound: -0.0008160, upper bound: 0.0008121
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 2, lower bound: -0.0007986, upper bound: 0.0008028
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.81
Output dim: 2, lower bound: -0.0007986, upper bound: 0.0008028

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000251, 0.0000252
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009415, 0.0009427
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011299, 0.0011313
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0083338, 0.0083443
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006346, 0.0006338
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006414, 0.0006406
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003116, 0.0003120
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021625, 0.0021598
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0017156, 0.0017135
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030857, 0.0030818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007785, upper bound: 0.0008115
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008152, upper bound: 0.0007752
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000252, 0.0000251
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009432, 0.0009408
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011318, 0.0011290
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0083483, 0.0083272
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006333, 0.0006349
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006401, 0.0006417
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003121, 0.0003113
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021581, 0.0021635
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0017121, 0.0017164
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030794, 0.0030872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006096, upper bound: 0.0006065
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006096, upper bound: 0.0006065
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000253, 0.0000253
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009470, 0.0009458
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011365, 0.0011350
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0083826, 0.0083718
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006367, 0.0006375
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006435, 0.0006444
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003134, 0.0003130
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021696, 0.0021724
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0017213, 0.0017235
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030959, 0.0030999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007496, upper bound: 0.0007769
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007718, upper bound: 0.0007532
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000257, 0.0000253
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009620, 0.0009475
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011544, 0.0011370
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0085145, 0.0083865
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006378, 0.0006476
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006447, 0.0006545
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003183, 0.0003136
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021734, 0.0022066
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0017243, 0.0017506
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0031013, 0.0031487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007860, upper bound: 0.0007904
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007862, upper bound: 0.0007903
time: 0.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 2, lower bound: -0.0007785, upper bound: 0.0008115
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 2, lower bound: -0.0008152, upper bound: 0.0007752
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 2, lower bound: -0.0006096, upper bound: 0.0006065
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 2, lower bound: -0.0006096, upper bound: 0.0006065
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 2, lower bound: -0.0007496, upper bound: 0.0007769
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 2, lower bound: -0.0007718, upper bound: 0.0007532
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 2, lower bound: -0.0007860, upper bound: 0.0007904
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 2, lower bound: -0.0007862, upper bound: 0.0007903

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000245, 0.0000248
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009162, 0.0009272
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0010995, 0.0011127
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0081099, 0.0082073
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006242, 0.0006168
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006309, 0.0006234
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003032, 0.0003069
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021270, 0.0021018
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016875, 0.0016674
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030351, 0.0029990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007717, upper bound: 0.0008065
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007735, upper bound: 0.0008047
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000248, 0.0000245
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009273, 0.0009174
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011128, 0.0011010
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0082079, 0.0081205
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006176, 0.0006243
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006242, 0.0006309
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003069, 0.0003036
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021045, 0.0021272
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016696, 0.0016876
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030029, 0.0030353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007657, upper bound: 0.0007490
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007915, upper bound: 0.0007344
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000249, 0.0000251
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009341, 0.0009390
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011210, 0.0011269
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0082681, 0.0083115
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006321, 0.0006288
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006389, 0.0006356
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003091, 0.0003108
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021540, 0.0021428
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0017089, 0.0017000
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030736, 0.0030575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007376, upper bound: 0.0007646
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007376, upper bound: 0.0007645
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000253, 0.0000249
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009470, 0.0009329
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011365, 0.0011195
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0083826, 0.0082574
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006280, 0.0006375
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006347, 0.0006444
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003134, 0.0003087
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021400, 0.0021724
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016977, 0.0017235
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030536, 0.0030999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007642, upper bound: 0.0007477
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007661, upper bound: 0.0007466
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000251, 0.0000248
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009408, 0.0009269
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011290, 0.0011124
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0083272, 0.0082047
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006240, 0.0006333
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006307, 0.0006401
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003113, 0.0003068
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021263, 0.0021581
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016869, 0.0017121
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030341, 0.0030794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007376, upper bound: 0.0007647
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007594, upper bound: 0.0007410
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000252, 0.0000247
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009427, 0.0009253
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011313, 0.0011104
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0083443, 0.0081902
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006229, 0.0006346
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006296, 0.0006414
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003120, 0.0003062
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0021226, 0.0021625
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016839, 0.0017156
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0030287, 0.0030857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007498, upper bound: 0.0007894
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007854, upper bound: 0.0007535
time: 0.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007717, upper bound: 0.0008065
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007735, upper bound: 0.0008047
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007657, upper bound: 0.0007490
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007915, upper bound: 0.0007344
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007376, upper bound: 0.0007646
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007376, upper bound: 0.0007645
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007642, upper bound: 0.0007477
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007661, upper bound: 0.0007466
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007376, upper bound: 0.0007647
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007594, upper bound: 0.0007410
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007498, upper bound: 0.0007894
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 2, lower bound: -0.0007854, upper bound: 0.0007535

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000239, 0.0000244
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0008966, 0.0009134
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0010760, 0.0010961
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0079365, 0.0080849
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006149, 0.0006036
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006215, 0.0006101
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0002967, 0.0003023
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020953, 0.0020568
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016623, 0.0016318
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029898, 0.0029349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007465, upper bound: 0.0007802
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007466, upper bound: 0.0007802
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000242, 0.0000242
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009064, 0.0009076
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0010877, 0.0010892
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0080230, 0.0080339
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006110, 0.0006102
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006175, 0.0006167
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003000, 0.0003004
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020821, 0.0020792
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016518, 0.0016496
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029709, 0.0029669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007482, upper bound: 0.0007786
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007484, upper bound: 0.0007786
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000248, 0.0000242
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009273, 0.0009048
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011128, 0.0010858
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0082079, 0.0080088
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006091, 0.0006243
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006156, 0.0006309
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003069, 0.0002994
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020756, 0.0021272
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016467, 0.0016876
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029617, 0.0030353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007090
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007089
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000245, 0.0000243
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009174, 0.0009106
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011010, 0.0010928
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0081205, 0.0080604
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006130, 0.0006176
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006196, 0.0006242
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003036, 0.0003014
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020889, 0.0021045
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016572, 0.0016696
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029807, 0.0030029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007090, upper bound: 0.0007637
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007217, upper bound: 0.0007402
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000248, 0.0000240
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009272, 0.0008996
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0011127, 0.0010795
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0082073, 0.0079624
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006056, 0.0006242
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006121, 0.0006309
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003069, 0.0002977
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020635, 0.0021270
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016371, 0.0016875
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029445, 0.0030351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007367, upper bound: 0.0007259
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007589, upper bound: 0.0007114
time: 0.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007465, upper bound: 0.0007802
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007466, upper bound: 0.0007802
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007482, upper bound: 0.0007786
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007484, upper bound: 0.0007786
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007089
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007090, upper bound: 0.0007637
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007217, upper bound: 0.0007402
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007367, upper bound: 0.0007259
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 2, lower bound: -0.0007589, upper bound: 0.0007114

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000234, 0.0000238
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0008775, 0.0008917
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0010530, 0.0010701
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0077667, 0.0078926
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006003, 0.0005907
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006067, 0.0005970
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0002904, 0.0002951
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020454, 0.0020128
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016228, 0.0015969
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029187, 0.0028721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007035, upper bound: 0.0007534
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007193, upper bound: 0.0007314
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000239, 0.0000239
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0008966, 0.0008942
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0010760, 0.0010731
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0079365, 0.0079150
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0006020, 0.0006036
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006084, 0.0006101
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0002967, 0.0002959
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020513, 0.0020568
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016274, 0.0016318
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029270, 0.0029349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005509, upper bound: 0.0005712
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005509, upper bound: 0.0005712
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000237, 0.0000237
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0008872, 0.0008869
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0010647, 0.0010643
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0078531, 0.0078500
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0005970, 0.0005973
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006034, 0.0006037
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0002936, 0.0002935
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020344, 0.0020352
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016140, 0.0016146
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029029, 0.0029041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005535, upper bound: 0.0005684
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005535, upper bound: 0.0005684
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0041536, -0.0041193, -0.0041536, -0.0041193, -0.0000242, 0.0000237
1: -0.0082066, -0.0069232, -0.0082066, -0.0069232, -0.0009064, 0.0008885
2: 0.9666153, 0.9681553, 0.9666153, 0.9681553, -0.0010877, 0.0010662
3: 0.0000651, 0.0114248, 0.0000651, 0.0114248, -0.0080230, 0.0078641
4: -0.0015620, -0.0006980, -0.0015620, -0.0006980, -0.0005981, 0.0006102
5: 0.0156917, 0.0165649, 0.0156917, 0.0165649, -0.0006045, 0.0006167
6: 0.0038496, 0.0042743, 0.0038496, 0.0042743, -0.0003000, 0.0002940
7: -0.0107391, -0.0077951, -0.0107391, -0.0077951, -0.0020380, 0.0020792
8: 0.0082093, 0.0105449, 0.0082093, 0.0105449, -0.0016169, 0.0016496
9: 0.0124898, 0.0166906, 0.0124898, 0.0166906, -0.0029081, 0.0029669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005535, upper bound: 0.0005684
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005535, upper bound: 0.0005684
time: 0.71 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 2, lower bound: -0.0007035, upper bound: 0.0007534
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 2, lower bound: -0.0007193, upper bound: 0.0007314
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 2, lower bound: -0.0005509, upper bound: 0.0005712
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 2, lower bound: -0.0005509, upper bound: 0.0005712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 2, lower bound: -0.0005535, upper bound: 0.0005684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 2, lower bound: -0.0005535, upper bound: 0.0005684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 2, lower bound: -0.0005535, upper bound: 0.0005684
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.82
Output dim: 2, lower bound: -0.0005535, upper bound: 0.0005684

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.10 + 64.62 = 67.72 seconds
