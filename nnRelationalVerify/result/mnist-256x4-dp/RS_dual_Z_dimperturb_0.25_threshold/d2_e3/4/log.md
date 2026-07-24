## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00217782


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0010069, 0.0010069)
1: (-0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0025819, 0.0025819)
2: (0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0020957, 0.0020957)
3: (-0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002835, 0.0002835)
4: (0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0015943, 0.0015943)
5: (-0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0016905, 0.0016905)
6: (-0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0008358, 0.0008358)
7: (-0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0015614, 0.0015614)
8: (-0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0002567, 0.0002567)
9: (1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0043105, 0.0043105)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.35 + 1.34 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0025073, upper bound: 0.0025073

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0024302, upper bound: 0.0023894
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0023894, upper bound: 0.0024303
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 9, lower bound: -0.0024302, upper bound: 0.0023894
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 9, lower bound: -0.0023894, upper bound: 0.0024303

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0009984, 0.0009973
1: -0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0025716, 0.0025781
2: 0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0020732, 0.0020877
3: -0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002744, 0.0002807
4: 0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0015841, 0.0015706
5: -0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0016750, 0.0016823
6: -0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0008190, 0.0008276
7: -0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0015506, 0.0015252
8: -0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0002525, 0.0002504
9: 1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0042952, 0.0042696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0023022, upper bound: 0.0022392
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0022704, upper bound: 0.0022815
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0010069, 0.0009984
1: -0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0025819, 0.0025716
2: 0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0020957, 0.0020732
3: -0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002835, 0.0002744
4: 0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0015706, 0.0015943
5: -0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0016905, 0.0016750
6: -0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0008358, 0.0008190
7: -0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0015252, 0.0015614
8: -0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0002504, 0.0002567
9: 1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0042696, 0.0043105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0022815, upper bound: 0.0022704
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0022392, upper bound: 0.0023022
time: 0.51 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 9, lower bound: -0.0023022, upper bound: 0.0022392
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 9, lower bound: -0.0022704, upper bound: 0.0022815
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 9, lower bound: -0.0022815, upper bound: 0.0022704
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 9, lower bound: -0.0022392, upper bound: 0.0023022

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0009668, 0.0009640
1: -0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0025053, 0.0025292
2: 0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0020366, 0.0020650
3: -0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002569, 0.0002696
4: 0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0015734, 0.0015550
5: -0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0016346, 0.0016548
6: -0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0008183, 0.0008274
7: -0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0015497, 0.0015162
8: -0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0002438, 0.0002351
9: 1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0042350, 0.0041730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0021997, upper bound: 0.0018580
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0019478, upper bound: 0.0021473
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0009650, 0.0009657
1: -0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0025226, 0.0025117
2: 0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0020505, 0.0020508
3: -0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002633, 0.0002632
4: 0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0015671, 0.0015599
5: -0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0016474, 0.0016420
6: -0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0008187, 0.0008268
7: -0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0015416, 0.0015244
8: -0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0002372, 0.0002406
9: 1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0042000, 0.0042094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021737, upper bound: 0.0019259
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0018874, upper bound: 0.0021770
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0009754, 0.0009650
1: -0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0025156, 0.0025226
2: 0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0020591, 0.0020505
3: -0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002664, 0.0002633
4: 0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0015599, 0.0015787
5: -0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0016502, 0.0016474
6: -0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0008351, 0.0008187
7: -0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0015244, 0.0015521
8: -0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0002406, 0.0002416
9: 1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0042094, 0.0042139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021770, upper bound: 0.0018874
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0019259, upper bound: 0.0021737
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0009736, 0.0009668
1: -0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0025329, 0.0025053
2: 0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0020730, 0.0020366
3: -0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002727, 0.0002569
4: 0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0015550, 0.0015836
5: -0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0016629, 0.0016346
6: -0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0008356, 0.0008183
7: -0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0015162, 0.0015603
8: -0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0002351, 0.0002471
9: 1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0041730, 0.0042503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021472, upper bound: 0.0019478
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0018580, upper bound: 0.0021997
time: 0.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 9, lower bound: -0.0021997, upper bound: 0.0018580
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 9, lower bound: -0.0019478, upper bound: 0.0021473
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 9, lower bound: -0.0021737, upper bound: 0.0019259
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 9, lower bound: -0.0018874, upper bound: 0.0021770
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 9, lower bound: -0.0021770, upper bound: 0.0018874
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 9, lower bound: -0.0019259, upper bound: 0.0021737
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 9, lower bound: -0.0021472, upper bound: 0.0019478
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 9, lower bound: -0.0018580, upper bound: 0.0021997

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0009204, 0.0009068
1: -0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0022508, 0.0023319
2: 0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0017764, 0.0018718
3: -0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002445, 0.0002568
4: 0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0013551, 0.0012773
5: -0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0013267, 0.0014160
6: -0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0006750, 0.0007133
7: -0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0014058, 0.0013298
8: -0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0002028, 0.0001994
9: 1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0037521, 0.0035296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0020693, upper bound: 0.0017827
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0021301, upper bound: 0.0017184
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0027712, -0.0012038, -0.0027712, -0.0012038, -0.0009162, 0.0009204
1: -0.0028842, 0.0011131, -0.0028842, 0.0011131, -0.0023349, 0.0022508
2: 0.0042408, 0.0074779, 0.0042408, 0.0074779, -0.0018812, 0.0017764
3: -0.0042971, -0.0038704, -0.0042971, -0.0038704, -0.0002597, 0.0002445
4: 0.0035640, 0.0063600, 0.0035640, 0.0063600, -0.0012773, 0.0013709
5: -0.0015200, 0.0013062, -0.0015200, 0.0013062, -0.0014262, 0.0013267
6: -0.0059250, -0.0043408, -0.0059250, -0.0043408, -0.0007230, 0.0006750
7: -0.0001424, 0.0025544, -0.0001424, 0.0025544, -0.0013298, 0.0014203
8: -0.0005060, -0.0001398, -0.0005060, -0.0001398, -0.0001994, 0.0002061
9: 1.0023179, 1.0090356, 1.0023179, 1.0090356, -0.0035296, 0.0037720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0017184, upper bound: 0.0021301
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0017827, upper bound: 0.0020693
time: 0.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 9, lower bound: -0.0020693, upper bound: 0.0017827
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 9, lower bound: -0.0021301, upper bound: 0.0017184
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 9, lower bound: -0.0017184, upper bound: 0.0021301
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 9, lower bound: -0.0017827, upper bound: 0.0020693

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.70 + 20.15 = 22.85 seconds
