## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.10584259


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084)
1: (-0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402)
2: (-0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730)
3: (-0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0732280, 0.0732280)
4: (-0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911)
5: (0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0467379, 0.0467379)
6: (-0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706)
7: (-0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233)
8: (0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2310944, 0.2310946)
9: (0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.38 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1351254, upper bound: 0.1351254

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1266827, upper bound: 0.1327468
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1327468, upper bound: 0.1266827
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 8, lower bound: -0.1266827, upper bound: 0.1327468
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 8, lower bound: -0.1327468, upper bound: 0.1266827

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0729730, 0.0728854
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0466376, 0.0465806
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2287908, 0.2293174
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1176821, upper bound: 0.1246022
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1171530, upper bound: 0.1245865
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0728854, 0.0729730
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465806, 0.0466376
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2293177, 0.2287908
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1245865, upper bound: 0.1171530
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1246022, upper bound: 0.1176821
time: 0.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 8, lower bound: -0.1176821, upper bound: 0.1246022
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 8, lower bound: -0.1171530, upper bound: 0.1245865
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 8, lower bound: -0.1245865, upper bound: 0.1171530
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 8, lower bound: -0.1246022, upper bound: 0.1176821

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0723062, 0.0722842
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0460996, 0.0460852
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2247338, 0.2248662
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1173803, upper bound: 0.0640724
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0644791, upper bound: 0.1243079
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0723718, 0.0721095
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0461422, 0.0459717
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2236843, 0.2252607
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1168506, upper bound: 0.0639877
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0644801, upper bound: 0.1242930
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0721095, 0.0723718
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0459716, 0.0461422
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2252607, 0.2236843
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1242930, upper bound: 0.0644801
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0639877, upper bound: 0.1168506
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0722842, 0.0723062
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0460853, 0.0460995
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2248664, 0.2247341
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 59

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1243079, upper bound: 0.0644791
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0640724, upper bound: 0.1173803
time: 0.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.1173803, upper bound: 0.0640724
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0644791, upper bound: 0.1243079
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.1168506, upper bound: 0.0639877
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0644801, upper bound: 0.1242930
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.1242930, upper bound: 0.0644801
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0639877, upper bound: 0.1168506
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.1243079, upper bound: 0.0644791
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 8, lower bound: -0.0640724, upper bound: 0.1173803

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0720512, 0.0722687
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465574, 0.0466988
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2251241, 0.2238176
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1143588, upper bound: 0.0591068
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0957692, upper bound: 0.0591738
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0723876, 0.0719635
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0467761, 0.0465004
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2232912, 0.2258382
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0595925, upper bound: 0.0973936
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0595402, upper bound: 0.1215973
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0720512, 0.0722687
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465574, 0.0466988
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2251241, 0.2238176
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1136864, upper bound: 0.0590518
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0960739, upper bound: 0.0591381
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0723876, 0.0719635
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0467761, 0.0465004
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2232912, 0.2258382
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0595790, upper bound: 0.0970913
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0595381, upper bound: 0.1215697
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0719636, 0.0723876
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465004, 0.0467761
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2258384, 0.2232909
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1215697, upper bound: 0.0595381
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0970913, upper bound: 0.0595790
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0722688, 0.0720513
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0466988, 0.0465574
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2238176, 0.2251241
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0591381, upper bound: 0.0960739
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0590518, upper bound: 0.1136864
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0719636, 0.0723876
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465004, 0.0467761
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2258384, 0.2232909
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1215973, upper bound: 0.0595402
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0973936, upper bound: 0.0595925
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0722688, 0.0720513
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0466988, 0.0465574
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2238176, 0.2251241
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0591738, upper bound: 0.0957692
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0591068, upper bound: 0.1143588
time: 0.49 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.1143588, upper bound: 0.0591068
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0957692, upper bound: 0.0591738
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0595925, upper bound: 0.0973936
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0595402, upper bound: 0.1215973
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.1136864, upper bound: 0.0590518
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0960739, upper bound: 0.0591381
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0595790, upper bound: 0.0970913
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0595381, upper bound: 0.1215697
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.1215697, upper bound: 0.0595381
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0970913, upper bound: 0.0595790
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0591381, upper bound: 0.0960739
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0590518, upper bound: 0.1136864
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.1215973, upper bound: 0.0595402
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0973936, upper bound: 0.0595925
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0591738, upper bound: 0.0957692
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 8, lower bound: -0.0591068, upper bound: 0.1143588

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0717952, 0.0720761
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464918, 0.0466744
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2238231, 0.2221355
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1132914, upper bound: 0.0581977
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1095890, upper bound: 0.0571843
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0721891, 0.0717075
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0467479, 0.0464348
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2216086, 0.2245016
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0578264, upper bound: 0.1161844
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0586196, upper bound: 0.1205714
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0717952, 0.0720761
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464918, 0.0466744
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2238231, 0.2221355
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1126413, upper bound: 0.0581377
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1096755, upper bound: 0.0571839
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0721891, 0.0717075
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0467479, 0.0464348
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2216086, 0.2245016
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0575801, upper bound: 0.1146767
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0586173, upper bound: 0.1205414
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0717075, 0.0721891
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464348, 0.0467479
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2245016, 0.2216089
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1205414, upper bound: 0.0586173
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1146767, upper bound: 0.0575801
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0720761, 0.0717952
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0466744, 0.0464919
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2221355, 0.2238228
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0571839, upper bound: 0.1096755
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0581377, upper bound: 0.1126413
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0717075, 0.0721891
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464348, 0.0467479
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2245016, 0.2216089
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1205714, upper bound: 0.0586196
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1161844, upper bound: 0.0578264
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0720761, 0.0717952
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0466744, 0.0464919
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2221355, 0.2238228
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 116
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 116

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0571843, upper bound: 0.1095890
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0581977, upper bound: 0.1132914
time: 0.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.1132914, upper bound: 0.0581977
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.1095890, upper bound: 0.0571843
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.0578264, upper bound: 0.1161844
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.0586196, upper bound: 0.1205714
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.1126413, upper bound: 0.0581377
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.1096755, upper bound: 0.0571839
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.0575801, upper bound: 0.1146767
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.0586173, upper bound: 0.1205414
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.1205414, upper bound: 0.0586173
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.1146767, upper bound: 0.0575801
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.0571839, upper bound: 0.1096755
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.0581377, upper bound: 0.1126413
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.1205714, upper bound: 0.0586196
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.1161844, upper bound: 0.0578264
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.0571843, upper bound: 0.1095890
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 8, lower bound: -0.0581977, upper bound: 0.1132914

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0730469, 0.0730999
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464984, 0.0465329
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2300701, 0.2297525
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0887959, upper bound: 0.0539766
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0747022, upper bound: 0.0540801
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0731875, 0.0729327
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465899, 0.0464243
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2290668, 0.2305968
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0886533, upper bound: 0.0528856
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0752928, upper bound: 0.0529918
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0730469, 0.0730999
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464984, 0.0465329
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2300701, 0.2297525
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0536384, upper bound: 0.0743562
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0536194, upper bound: 0.0897991
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0731875, 0.0729327
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465899, 0.0464243
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2290668, 0.2305968
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0544694, upper bound: 0.0719691
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0544043, upper bound: 0.0897637
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0730469, 0.0730999
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464984, 0.0465329
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2300701, 0.2297525
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0883091, upper bound: 0.0539214
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0720074, upper bound: 0.0539461
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0731875, 0.0729327
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465899, 0.0464243
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2290668, 0.2305968
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0883050, upper bound: 0.0529295
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0740572, upper bound: 0.0530038
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0730469, 0.0730999
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464984, 0.0465329
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2300701, 0.2297525
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0534167, upper bound: 0.0753303
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0532471, upper bound: 0.0898453
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0731875, 0.0729327
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465899, 0.0464243
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2290668, 0.2305968
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0544689, upper bound: 0.0746734
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0543981, upper bound: 0.0898459
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0729328, 0.0731875
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464243, 0.0465899
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2305965, 0.2290671
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0898459, upper bound: 0.0543981
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0746734, upper bound: 0.0544689
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0730997, 0.0730469
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465329, 0.0464985
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2297525, 0.2300704
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0898453, upper bound: 0.0532471
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0753303, upper bound: 0.0534167
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0729328, 0.0731875
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464243, 0.0465899
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2305965, 0.2290671
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0530038, upper bound: 0.0740572
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0529295, upper bound: 0.0883050
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0730997, 0.0730469
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465329, 0.0464985
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2297525, 0.2300704
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0539461, upper bound: 0.0720074
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0539214, upper bound: 0.0883091
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0729328, 0.0731875
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464243, 0.0465899
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2305965, 0.2290671
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0897637, upper bound: 0.0544043
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0719691, upper bound: 0.0544694
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0730997, 0.0730469
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465329, 0.0464985
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2297525, 0.2300704
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0897991, upper bound: 0.0536194
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0743562, upper bound: 0.0536384
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0729328, 0.0731875
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0464243, 0.0465899
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2305965, 0.2290671
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0529918, upper bound: 0.0752928
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0528856, upper bound: 0.0886533
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0463111, 0.0704973, -0.0463111, 0.0704973, -0.1168084, 0.1168084
1: -0.0163713, 0.0161688, -0.0163713, 0.0161688, -0.0325402, 0.0325402
2: -0.0049708, 0.0409022, -0.0049708, 0.0409022, -0.0458730, 0.0458730
3: -0.0067138, 0.0746847, -0.0067138, 0.0746847, -0.0730997, 0.0730469
4: -0.0256890, -0.0011980, -0.0256890, -0.0011980, -0.0244911, 0.0244911
5: 0.0029544, 0.0520040, 0.0029544, 0.0520040, -0.0465329, 0.0464985
6: -0.0368911, 0.0609795, -0.0368911, 0.0609795, -0.0978706, 0.0978706
7: -0.0178028, 0.0135206, -0.0178028, 0.0135206, -0.0313233, 0.0313233
8: 0.6693820, 0.9487947, 0.6693820, 0.9487947, -0.2297525, 0.2300704
9: 0.0472139, 0.0961027, 0.0472139, 0.0961027, -0.0488888, 0.0488888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 154
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 29

### Candidate
type: RSZ, layer: 3, pos: 59

### Candidate
type: RSZ, layer: 3, pos: 71

### Candidate
type: RSZ, layer: 3, pos: 154

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0540801, upper bound: 0.0747022
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0539766, upper bound: 0.0887959
time: 0.51 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0887959, upper bound: 0.0539766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0747022, upper bound: 0.0540801
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0886533, upper bound: 0.0528856
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0752928, upper bound: 0.0529918
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0536384, upper bound: 0.0743562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0536194, upper bound: 0.0897991
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0544694, upper bound: 0.0719691
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0544043, upper bound: 0.0897637
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0883091, upper bound: 0.0539214
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0720074, upper bound: 0.0539461
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0883050, upper bound: 0.0529295
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0740572, upper bound: 0.0530038
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0534167, upper bound: 0.0753303
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0532471, upper bound: 0.0898453
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0544689, upper bound: 0.0746734
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0543981, upper bound: 0.0898459
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0898459, upper bound: 0.0543981
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0746734, upper bound: 0.0544689
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0898453, upper bound: 0.0532471
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0753303, upper bound: 0.0534167
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0530038, upper bound: 0.0740572
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0529295, upper bound: 0.0883050
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0539461, upper bound: 0.0720074
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0539214, upper bound: 0.0883091
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0897637, upper bound: 0.0544043
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0719691, upper bound: 0.0544694
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0897991, upper bound: 0.0536194
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0743562, upper bound: 0.0536384
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0529918, upper bound: 0.0752928
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0528856, upper bound: 0.0886533
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0540801, upper bound: 0.0747022
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.26
Output dim: 8, lower bound: -0.0539766, upper bound: 0.0887959

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.59 + 87.63 = 90.22 seconds
