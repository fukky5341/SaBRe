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
Threshold: 0.0016185600000000002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013854, 0.0013854)
1: (-0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035157, 0.0035157)
2: (0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021812, 0.0021812)
3: (0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040728, 0.0040728)
4: (-0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035761, 0.0035761)
5: (0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013545, 0.0013545)
6: (0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051689, 0.0051689)
7: (0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036170, 0.0036170)
8: (-0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038780, 0.0038780)
9: (-0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025616, 0.0025616)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 2.81 = 4.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0020232, upper bound: 0.0020232

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019127, upper bound: 0.0019127
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019127, upper bound: 0.0019127
time: 1.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.18
Output dim: 7, lower bound: -0.0019127, upper bound: 0.0019127
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.18
Output dim: 7, lower bound: -0.0019127, upper bound: 0.0019127

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013796, 0.0013851
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035010, 0.0035150
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021721, 0.0021807
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040720, 0.0040558
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035612, 0.0035754
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013489, 0.0013542
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051679, 0.0051473
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036162, 0.0036019
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038772, 0.0038618
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025509, 0.0025611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
time: 1.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013854, 0.0013796
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035157, 0.0035010
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021812, 0.0021721
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040558, 0.0040728
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035761, 0.0035612
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013545, 0.0013489
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051473, 0.0051689
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036019, 0.0036170
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038618, 0.0038780
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025616, 0.0025509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019115, upper bound: 0.0019116
time: 1.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.00
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.00
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.00
Output dim: 7, lower bound: -0.0019116, upper bound: 0.0019116
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.00
Output dim: 7, lower bound: -0.0019115, upper bound: 0.0019116

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013800, 0.0013857
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035020, 0.0035164
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021727, 0.0021816
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040736, 0.0040569
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035621, 0.0035768
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013492, 0.0013548
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051699, 0.0051487
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036177, 0.0036028
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038787, 0.0038628
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025516, 0.0025621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018652, upper bound: 0.0017939
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018651
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013802, 0.0013855
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035025, 0.0035160
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021730, 0.0021813
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040731, 0.0040575
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035626, 0.0035763
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013494, 0.0013546
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051693, 0.0051494
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036172, 0.0036033
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038782, 0.0038633
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025520, 0.0025618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018651, upper bound: 0.0017939
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018652
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013857, 0.0013802
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035165, 0.0035025
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021816, 0.0021730
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040575, 0.0040737
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035768, 0.0035626
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013548, 0.0013494
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051494, 0.0051700
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036033, 0.0036177
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038633, 0.0038788
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025621, 0.0025520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018652, upper bound: 0.0017939
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018651
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013859, 0.0013800
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035170, 0.0035020
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021819, 0.0021727
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040569, 0.0040742
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035773, 0.0035621
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013550, 0.0013492
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051487, 0.0051707
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036028, 0.0036182
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038628, 0.0038793
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025625, 0.0025516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018651, upper bound: 0.0017939
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018652
time: 1.22 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 7, lower bound: -0.0018652, upper bound: 0.0017939
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018651
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 7, lower bound: -0.0018651, upper bound: 0.0017939
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018652
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 7, lower bound: -0.0018652, upper bound: 0.0017939
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018651
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 7, lower bound: -0.0018651, upper bound: 0.0017939
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.62
Output dim: 7, lower bound: -0.0017939, upper bound: 0.0018652

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013154, 0.0013361
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033379, 0.0033906
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020709, 0.0021035
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039278, 0.0038668
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033952, 0.0034488
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012860, 0.0013063
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049849, 0.0049075
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034882, 0.0034340
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037399, 0.0036818
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024320, 0.0024704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013311, 0.0013210
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033780, 0.0033523
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020957, 0.0020798
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038835, 0.0039132
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034360, 0.0034099
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013015, 0.0012916
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049287, 0.0049664
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034488, 0.0034752
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036977, 0.0037260
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024612, 0.0024425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013156, 0.0013360
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033384, 0.0033902
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020712, 0.0021033
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039274, 0.0038674
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033957, 0.0034484
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012862, 0.0013062
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049844, 0.0049082
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034878, 0.0034345
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037395, 0.0036824
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024324, 0.0024701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013314, 0.0013209
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033785, 0.0033519
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020960, 0.0020795
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038830, 0.0039138
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034365, 0.0034094
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013017, 0.0012914
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049280, 0.0049672
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034484, 0.0034758
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036972, 0.0037266
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024616, 0.0024422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013211, 0.0013314
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033526, 0.0033785
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020799, 0.0020960
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039138, 0.0038838
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034101, 0.0034365
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012917, 0.0013017
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049672, 0.0049290
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034758, 0.0034491
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037266, 0.0036980
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024427, 0.0024616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013369, 0.0013156
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033926, 0.0033384
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021048, 0.0020712
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038674, 0.0039302
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034509, 0.0033957
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013071, 0.0012862
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049082, 0.0049879
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034345, 0.0034903
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036824, 0.0037421
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024719, 0.0024324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013213, 0.0013311
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033530, 0.0033780
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020802, 0.0020957
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0039132, 0.0038843
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034106, 0.0034360
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012918, 0.0013015
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049664, 0.0049297
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034752, 0.0034496
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0037260, 0.0036985
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024431, 0.0024612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
time: 1.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013371, 0.0013154
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033931, 0.0033379
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021051, 0.0020709
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038668, 0.0039308
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034514, 0.0033952
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013073, 0.0012860
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049075, 0.0049887
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034340, 0.0034908
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036818, 0.0037427
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024723, 0.0024320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
time: 1.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017870, upper bound: 0.0017138
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017140, upper bound: 0.0017866
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017866, upper bound: 0.0017140
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.06
Output dim: 7, lower bound: -0.0017138, upper bound: 0.0017870

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012857, 0.0013073
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032625, 0.0033175
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020241, 0.0020582
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038431, 0.0037795
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033186, 0.0033744
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012570, 0.0012781
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048774, 0.0047967
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034130, 0.0033565
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036593, 0.0035987
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023771, 0.0024172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
time: 1.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012866, 0.0013071
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032648, 0.0033169
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020255, 0.0020578
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038425, 0.0037821
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033209, 0.0033739
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012579, 0.0012779
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048766, 0.0048000
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034124, 0.0033588
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036587, 0.0036012
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023788, 0.0024167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013019, 0.0012922
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033037, 0.0032792
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020496, 0.0020344
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037988, 0.0038271
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033604, 0.0033355
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012728, 0.0012634
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048212, 0.0048571
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033736, 0.0033988
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036171, 0.0036440
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024071, 0.0023893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013023, 0.0012917
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033049, 0.0032778
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020504, 0.0020336
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037972, 0.0038285
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033616, 0.0033341
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012733, 0.0012629
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048191, 0.0048589
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033722, 0.0034000
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036155, 0.0036454
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024080, 0.0023882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012858, 0.0013072
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032629, 0.0033171
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020243, 0.0020579
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038427, 0.0037799
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033189, 0.0033740
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012571, 0.0012780
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048769, 0.0047971
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034126, 0.0033568
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036589, 0.0035990
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023774, 0.0024169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012867, 0.0013069
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032653, 0.0033165
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020258, 0.0020576
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038420, 0.0037827
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033214, 0.0033735
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012580, 0.0012778
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048760, 0.0048007
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034120, 0.0033593
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036582, 0.0036017
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023791, 0.0024165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013020, 0.0012921
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033040, 0.0032788
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020498, 0.0020342
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037983, 0.0038275
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033607, 0.0033351
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012730, 0.0012632
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048205, 0.0048576
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033732, 0.0033991
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036166, 0.0036444
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024073, 0.0023890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017125
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
time: 1.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013025, 0.0012916
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033054, 0.0032775
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020507, 0.0020334
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037968, 0.0038291
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033621, 0.0033338
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012735, 0.0012627
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048187, 0.0048597
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033719, 0.0034006
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036152, 0.0036459
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024084, 0.0023880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017126
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012914, 0.0013025
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032772, 0.0033054
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020332, 0.0020507
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038291, 0.0037965
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033334, 0.0033621
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012626, 0.0012735
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048597, 0.0048182
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034006, 0.0033715
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036459, 0.0036148
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023878, 0.0024084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012923, 0.0013020
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032795, 0.0033040
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020346, 0.0020498
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038275, 0.0037991
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033358, 0.0033607
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012635, 0.0012730
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048576, 0.0048215
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033991, 0.0033739
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036444, 0.0036173
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023895, 0.0024073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013076, 0.0012867
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033183, 0.0032653
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020587, 0.0020258
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037827, 0.0038441
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033753, 0.0033214
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012785, 0.0012580
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048007, 0.0048787
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033593, 0.0034138
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036017, 0.0036602
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024178, 0.0023791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013081, 0.0012858
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033195, 0.0032629
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020594, 0.0020243
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037799, 0.0038455
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033765, 0.0033189
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012789, 0.0012571
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047971, 0.0048804
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033568, 0.0034151
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035990, 0.0036615
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024186, 0.0023774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012915, 0.0013023
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032775, 0.0033049
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020334, 0.0020504
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038285, 0.0037968
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033338, 0.0033616
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012627, 0.0012733
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048589, 0.0048186
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034000, 0.0033719
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036454, 0.0036152
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023880, 0.0024080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012925, 0.0013019
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032799, 0.0033037
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020349, 0.0020496
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038271, 0.0037997
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033363, 0.0033604
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012637, 0.0012728
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048571, 0.0048223
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033988, 0.0033744
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036440, 0.0036179
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023898, 0.0024071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016431
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013078, 0.0012866
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033186, 0.0032648
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020589, 0.0020255
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037821, 0.0038445
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033756, 0.0033209
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012786, 0.0012579
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048000, 0.0048792
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033588, 0.0034142
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036012, 0.0036606
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024180, 0.0023788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017125
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
time: 1.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013083, 0.0012857
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033200, 0.0032625
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020598, 0.0020241
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037795, 0.0038461
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033770, 0.0033186
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012791, 0.0012570
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047967, 0.0048812
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033565, 0.0034156
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035987, 0.0036621
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024190, 0.0023771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017126
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
time: 1.68 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017125
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017126
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0016432
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017125, upper bound: 0.0016681
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016696, upper bound: 0.0017126
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016431, upper bound: 0.0017403
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016430
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017403, upper bound: 0.0016431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0017126, upper bound: 0.0016696
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017125
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016681, upper bound: 0.0017126
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0017411

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012062, 0.0012360
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030608, 0.0031365
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018989, 0.0019459
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036334, 0.0035458
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031133, 0.0031903
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011792, 0.0012084
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046113, 0.0045001
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032268, 0.0031489
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034596, 0.0033761
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022301, 0.0022853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012162, 0.0012278
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030862, 0.0031157
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019147, 0.0019330
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036094, 0.0035752
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031392, 0.0031692
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011890, 0.0012004
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045808, 0.0045374
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032054, 0.0031750
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034367, 0.0034041
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022486, 0.0022702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012071, 0.0012356
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030631, 0.0031354
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019003, 0.0019452
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036322, 0.0035484
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031157, 0.0031892
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011801, 0.0012080
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046098, 0.0045034
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032257, 0.0031513
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034584, 0.0033787
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022318, 0.0022845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012178, 0.0012276
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030903, 0.0031152
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019172, 0.0019327
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036088, 0.0035800
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031434, 0.0031687
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011906, 0.0012002
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045800, 0.0045435
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032049, 0.0031793
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034361, 0.0034087
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022516, 0.0022698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
time: 1.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012224, 0.0012221
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031019, 0.0031013
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019244, 0.0019240
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035927, 0.0035934
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031552, 0.0031545
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011951, 0.0011948
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045595, 0.0045605
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031905, 0.0031912
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034208, 0.0034215
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022601, 0.0022596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012317, 0.0012127
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031256, 0.0030775
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019391, 0.0019093
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035651, 0.0036209
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031793, 0.0031303
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012042, 0.0011857
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045246, 0.0045954
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031661, 0.0032156
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033945, 0.0034477
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022774, 0.0022423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012228, 0.0012209
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031031, 0.0030983
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019252, 0.0019222
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035892, 0.0035948
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031564, 0.0031515
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011956, 0.0011937
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045552, 0.0045623
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031875, 0.0031925
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034175, 0.0034228
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022610, 0.0022574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012326, 0.0012122
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031279, 0.0030761
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019406, 0.0019084
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035635, 0.0036235
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031816, 0.0031289
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012051, 0.0011851
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045225, 0.0045987
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031646, 0.0032180
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033930, 0.0034502
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022790, 0.0022413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012063, 0.0012358
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030611, 0.0031361
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018991, 0.0019456
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036330, 0.0035461
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031137, 0.0031899
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011794, 0.0012082
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046107, 0.0045005
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032263, 0.0031492
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034592, 0.0033765
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022304, 0.0022850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012164, 0.0012277
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030868, 0.0031154
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019150, 0.0019328
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036090, 0.0035759
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031398, 0.0031688
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011893, 0.0012003
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045803, 0.0045383
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032051, 0.0031756
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034363, 0.0034048
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022491, 0.0022699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012072, 0.0012353
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030636, 0.0031348
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019006, 0.0019448
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036315, 0.0035490
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031162, 0.0031886
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011803, 0.0012078
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046089, 0.0045041
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032251, 0.0031518
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034578, 0.0033792
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022321, 0.0022841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012180, 0.0012274
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030909, 0.0031148
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019176, 0.0019324
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036083, 0.0035807
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031440, 0.0031683
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011909, 0.0012000
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045794, 0.0045444
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032045, 0.0031799
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034357, 0.0034094
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022521, 0.0022695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012225, 0.0012219
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031023, 0.0031008
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019247, 0.0019237
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035921, 0.0035938
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031555, 0.0031540
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011952, 0.0011947
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045588, 0.0045610
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031901, 0.0031916
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034202, 0.0034219
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022603, 0.0022593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012320, 0.0012126
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031263, 0.0030770
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019396, 0.0019090
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035646, 0.0036216
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031799, 0.0031299
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012045, 0.0011855
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045239, 0.0045963
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031656, 0.0032163
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033941, 0.0034484
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022778, 0.0022420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012230, 0.0012207
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031037, 0.0030977
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019255, 0.0019218
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035886, 0.0035954
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031569, 0.0031509
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011958, 0.0011935
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045543, 0.0045631
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031869, 0.0031930
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034169, 0.0034234
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022614, 0.0022570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016255, upper bound: 0.0016650
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016257, upper bound: 0.0016632
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012328, 0.0012121
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031285, 0.0030758
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019409, 0.0019082
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035631, 0.0036242
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031822, 0.0031286
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012053, 0.0011850
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045220, 0.0045996
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031643, 0.0032186
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033926, 0.0034508
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022795, 0.0022410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015981, upper bound: 0.0016945
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015995, upper bound: 0.0016939
time: 1.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012118, 0.0012328
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030752, 0.0031285
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019079, 0.0019409
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036242, 0.0035625
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031280, 0.0031822
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011848, 0.0012053
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045996, 0.0045213
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032186, 0.0031638
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034508, 0.0033921
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022407, 0.0022795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012218, 0.0012230
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031006, 0.0031037
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019236, 0.0019255
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035954, 0.0035919
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031538, 0.0031569
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011946, 0.0011958
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045631, 0.0045586
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031930, 0.0031899
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034234, 0.0034201
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022591, 0.0022614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012127, 0.0012320
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030775, 0.0031263
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019093, 0.0019396
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036216, 0.0035652
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031304, 0.0031799
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011857, 0.0012045
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045963, 0.0045246
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032163, 0.0031661
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034484, 0.0033946
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022423, 0.0022778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016939, upper bound: 0.0015995
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016945, upper bound: 0.0015981
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012235, 0.0012225
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031048, 0.0031023
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019262, 0.0019247
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035938, 0.0035967
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031581, 0.0031555
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011962, 0.0011952
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045610, 0.0045647
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031916, 0.0031942
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034219, 0.0034246
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022622, 0.0022603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016632, upper bound: 0.0016257
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016255
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012281, 0.0012180
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031164, 0.0030909
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019334, 0.0019176
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035807, 0.0036102
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031699, 0.0031440
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012007, 0.0011909
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045444, 0.0045818
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031799, 0.0032061
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034094, 0.0034374
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022706, 0.0022521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012374, 0.0012072
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031401, 0.0030636
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019481, 0.0019006
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035490, 0.0036376
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031940, 0.0031162
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012098, 0.0011803
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045041, 0.0046166
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031518, 0.0032305
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033792, 0.0034636
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022879, 0.0022321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
time: 1.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012285, 0.0012164
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031176, 0.0030868
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019342, 0.0019150
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035759, 0.0036116
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031711, 0.0031398
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012011, 0.0011893
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045383, 0.0045835
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031756, 0.0032073
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034048, 0.0034388
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022715, 0.0022491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016650
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016269, upper bound: 0.0016631
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012383, 0.0012063
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031423, 0.0030611
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019495, 0.0018991
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035461, 0.0036403
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031963, 0.0031137
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012107, 0.0011794
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045005, 0.0046200
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031492, 0.0032328
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033765, 0.0034661
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022895, 0.0022304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015979, upper bound: 0.0016933
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015994, upper bound: 0.0016918
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012120, 0.0012326
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030756, 0.0031279
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019081, 0.0019406
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036235, 0.0035629
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031284, 0.0031816
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011849, 0.0012051
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045987, 0.0045218
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032180, 0.0031641
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034502, 0.0033924
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022409, 0.0022790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012221, 0.0012228
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031012, 0.0031031
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019240, 0.0019252
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035948, 0.0035926
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031545, 0.0031564
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011948, 0.0011956
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045623, 0.0045595
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031925, 0.0031905
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034228, 0.0034207
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022596, 0.0022610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012129, 0.0012317
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030780, 0.0031256
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019096, 0.0019391
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036209, 0.0035657
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031309, 0.0031793
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011859, 0.0012042
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045954, 0.0045254
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032156, 0.0031666
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034477, 0.0033951
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022427, 0.0022774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016918, upper bound: 0.0015994
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016933, upper bound: 0.0015979
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012237, 0.0012224
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031054, 0.0031019
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019266, 0.0019244
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035934, 0.0035975
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031587, 0.0031552
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011964, 0.0011951
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045605, 0.0045656
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031912, 0.0031948
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034215, 0.0034253
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022626, 0.0022601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016631, upper bound: 0.0016269
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016650, upper bound: 0.0016269
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012282, 0.0012178
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031167, 0.0030903
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019336, 0.0019172
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035800, 0.0036106
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031702, 0.0031434
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012008, 0.0011906
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045435, 0.0045823
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031793, 0.0032064
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034087, 0.0034378
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022709, 0.0022516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 4.08 + 596.63 = 600.70 seconds
