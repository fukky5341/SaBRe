## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 5.355e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0037588, -0.0031146, -0.0037588, -0.0031146, -0.0006442, 0.0006442)
1: (0.0059507, 0.0063894, 0.0059507, 0.0063894, -0.0004386, 0.0004386)
2: (0.0110470, 0.0122664, 0.0110470, 0.0122664, -0.0009890, 0.0009890)
3: (-0.0035614, -0.0030313, -0.0035614, -0.0030313, -0.0005301, 0.0005301)
4: (0.0049426, 0.0051202, 0.0049426, 0.0051202, -0.0000970, 0.0000970)
5: (-0.0014920, -0.0010797, -0.0014920, -0.0010797, -0.0004123, 0.0004123)
6: (-0.0056040, -0.0054103, -0.0056040, -0.0054103, -0.0001938, 0.0001938)
7: (-0.0030393, -0.0026827, -0.0030393, -0.0026827, -0.0003566, 0.0003566)
8: (-0.0025910, -0.0017823, -0.0025910, -0.0017823, -0.0008087, 0.0008087)
9: (1.0004613, 1.0005296, 1.0004613, 1.0005296, -0.0000683, 0.0000683)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.20 + 1.23 = 2.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0000630, upper bound: 0.0000630

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000626, upper bound: 0.0000626
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000625, upper bound: 0.0000626
time: 0.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 9, lower bound: -0.0000626, upper bound: 0.0000626
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.14
Output dim: 9, lower bound: -0.0000625, upper bound: 0.0000626

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037588, -0.0031146, -0.0037588, -0.0031146, -0.0006442, 0.0006442
1: 0.0059507, 0.0063894, 0.0059507, 0.0063894, -0.0004386, 0.0004386
2: 0.0110470, 0.0122664, 0.0110470, 0.0122664, -0.0009683, 0.0009762
3: -0.0035614, -0.0030313, -0.0035614, -0.0030313, -0.0005301, 0.0005301
4: 0.0049426, 0.0051202, 0.0049426, 0.0051202, -0.0000825, 0.0000786
5: -0.0014920, -0.0010797, -0.0014920, -0.0010797, -0.0004123, 0.0004123
6: -0.0056040, -0.0054103, -0.0056040, -0.0054103, -0.0001938, 0.0001938
7: -0.0030393, -0.0026827, -0.0030393, -0.0026827, -0.0003566, 0.0003566
8: -0.0025910, -0.0017823, -0.0025910, -0.0017823, -0.0008087, 0.0008087
9: 1.0004613, 1.0005296, 1.0004613, 1.0005296, -0.0000683, 0.0000683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000574, upper bound: 0.0000574
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000574, upper bound: 0.0000574
time: 0.42 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037588, -0.0031146, -0.0037588, -0.0031146, -0.0006442, 0.0006442
1: 0.0059507, 0.0063894, 0.0059507, 0.0063894, -0.0004386, 0.0004386
2: 0.0110470, 0.0122664, 0.0110470, 0.0122664, -0.0009762, 0.0009890
3: -0.0035614, -0.0030313, -0.0035614, -0.0030313, -0.0005301, 0.0005301
4: 0.0049426, 0.0051202, 0.0049426, 0.0051202, -0.0000970, 0.0000825
5: -0.0014920, -0.0010797, -0.0014920, -0.0010797, -0.0004123, 0.0004123
6: -0.0056040, -0.0054103, -0.0056040, -0.0054103, -0.0001938, 0.0001938
7: -0.0030393, -0.0026827, -0.0030393, -0.0026827, -0.0003566, 0.0003566
8: -0.0025910, -0.0017823, -0.0025910, -0.0017823, -0.0008087, 0.0008087
9: 1.0004613, 1.0005296, 1.0004613, 1.0005296, -0.0000683, 0.0000683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000574, upper bound: 0.0000574
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0000574, upper bound: 0.0000574
time: 0.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 9, lower bound: -0.0000574, upper bound: 0.0000574
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 9, lower bound: -0.0000574, upper bound: 0.0000574
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 9, lower bound: -0.0000574, upper bound: 0.0000574
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 9, lower bound: -0.0000574, upper bound: 0.0000574

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037588, -0.0031146, -0.0037588, -0.0031146, -0.0006442, 0.0006442
1: 0.0059507, 0.0063894, 0.0059507, 0.0063894, -0.0004386, 0.0004386
2: 0.0110470, 0.0122664, 0.0110470, 0.0122664, -0.0009642, 0.0009753
3: -0.0035614, -0.0030313, -0.0035614, -0.0030313, -0.0005296, 0.0005301
4: 0.0049426, 0.0051202, 0.0049426, 0.0051202, -0.0000825, 0.0000786
5: -0.0014920, -0.0010797, -0.0014920, -0.0010797, -0.0004123, 0.0004123
6: -0.0056040, -0.0054103, -0.0056040, -0.0054103, -0.0001938, 0.0001938
7: -0.0030393, -0.0026827, -0.0030393, -0.0026827, -0.0003566, 0.0003566
8: -0.0025910, -0.0017823, -0.0025910, -0.0017823, -0.0008087, 0.0008087
9: 1.0004613, 1.0005296, 1.0004613, 1.0005296, -0.0000683, 0.0000683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037588, -0.0031146, -0.0037588, -0.0031146, -0.0006442, 0.0006442
1: 0.0059507, 0.0063894, 0.0059507, 0.0063894, -0.0004386, 0.0004386
2: 0.0110470, 0.0122664, 0.0110470, 0.0122664, -0.0009683, 0.0009721
3: -0.0035614, -0.0030313, -0.0035614, -0.0030313, -0.0005301, 0.0005292
4: 0.0049426, 0.0051202, 0.0049426, 0.0051202, -0.0000824, 0.0000786
5: -0.0014920, -0.0010797, -0.0014920, -0.0010797, -0.0004123, 0.0004123
6: -0.0056040, -0.0054103, -0.0056040, -0.0054103, -0.0001938, 0.0001938
7: -0.0030393, -0.0026827, -0.0030393, -0.0026827, -0.0003566, 0.0003566
8: -0.0025910, -0.0017823, -0.0025910, -0.0017823, -0.0008087, 0.0008087
9: 1.0004613, 1.0005296, 1.0004613, 1.0005296, -0.0000683, 0.0000683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0037588, -0.0031146, -0.0037588, -0.0031146, -0.0006442, 0.0006442
1: 0.0059507, 0.0063894, 0.0059507, 0.0063894, -0.0004386, 0.0004386
2: 0.0110470, 0.0122664, 0.0110470, 0.0122664, -0.0009721, 0.0009881
3: -0.0035614, -0.0030313, -0.0035614, -0.0030313, -0.0005301, 0.0005301
4: 0.0049426, 0.0051202, 0.0049426, 0.0051202, -0.0000970, 0.0000824
5: -0.0014920, -0.0010797, -0.0014920, -0.0010797, -0.0004123, 0.0004123
6: -0.0056040, -0.0054103, -0.0056040, -0.0054103, -0.0001938, 0.0001938
7: -0.0030393, -0.0026827, -0.0030393, -0.0026827, -0.0003566, 0.0003566
8: -0.0025910, -0.0017823, -0.0025910, -0.0017823, -0.0008087, 0.0008087
9: 1.0004613, 1.0005296, 1.0004613, 1.0005296, -0.0000683, 0.0000683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0037588, -0.0031146, -0.0037588, -0.0031146, -0.0006442, 0.0006442
1: 0.0059507, 0.0063894, 0.0059507, 0.0063894, -0.0004386, 0.0004386
2: 0.0110470, 0.0122664, 0.0110470, 0.0122664, -0.0009762, 0.0009849
3: -0.0035614, -0.0030313, -0.0035614, -0.0030313, -0.0005301, 0.0005296
4: 0.0049426, 0.0051202, 0.0049426, 0.0051202, -0.0000970, 0.0000825
5: -0.0014920, -0.0010797, -0.0014920, -0.0010797, -0.0004123, 0.0004123
6: -0.0056040, -0.0054103, -0.0056040, -0.0054103, -0.0001938, 0.0001938
7: -0.0030393, -0.0026827, -0.0030393, -0.0026827, -0.0003566, 0.0003566
8: -0.0025910, -0.0017823, -0.0025910, -0.0017823, -0.0008087, 0.0008087
9: 1.0004613, 1.0005296, 1.0004613, 1.0005296, -0.0000683, 0.0000683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
time: 0.43 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 9, lower bound: -0.0000481, upper bound: 0.0000481

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.44 + 14.08 = 16.52 seconds
