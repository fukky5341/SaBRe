## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0028484


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0027761, -0.0021927, -0.0027761, -0.0021927, -0.0002234, 0.0002234)
1: (0.0238819, 0.0269550, 0.0238819, 0.0269550, -0.0012450, 0.0012450)
2: (0.0233083, 0.0254638, 0.0233083, 0.0254638, -0.0008481, 0.0008481)
3: (0.0111606, 0.0135748, 0.0111606, 0.0135748, -0.0010549, 0.0010549)
4: (-0.0138555, -0.0111989, -0.0138555, -0.0111989, -0.0011216, 0.0011216)
5: (0.0184328, 0.0214257, 0.0184328, 0.0214257, -0.0012953, 0.0012953)
6: (0.0090552, 0.0114284, 0.0090552, 0.0114284, -0.0010213, 0.0010213)
7: (-0.0186169, -0.0161470, -0.0186169, -0.0161470, -0.0009873, 0.0009873)
8: (0.0131673, 0.0154528, 0.0131673, 0.0154528, -0.0010315, 0.0010315)
9: (0.9183307, 0.9296775, 0.9183307, 0.9296775, -0.0048804, 0.0048804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.22 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0034800, upper bound: 0.0034800

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0033083, upper bound: 0.0033217
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0033217, upper bound: 0.0033083
time: 0.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 9, lower bound: -0.0033083, upper bound: 0.0033217
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 9, lower bound: -0.0033217, upper bound: 0.0033083

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0027761, -0.0021927, -0.0027761, -0.0021927, -0.0002231, 0.0002230
1: 0.0238819, 0.0269550, 0.0238819, 0.0269550, -0.0011473, 0.0011377
2: 0.0233083, 0.0254638, 0.0233083, 0.0254638, -0.0008095, 0.0008041
3: 0.0111606, 0.0135748, 0.0111606, 0.0135748, -0.0009769, 0.0009676
4: -0.0138555, -0.0111989, -0.0138555, -0.0111989, -0.0010716, 0.0010796
5: 0.0184328, 0.0214257, 0.0184328, 0.0214257, -0.0012352, 0.0012245
6: 0.0090552, 0.0114284, 0.0090552, 0.0114284, -0.0009711, 0.0009620
7: -0.0186169, -0.0161470, -0.0186169, -0.0161470, -0.0009445, 0.0009513
8: 0.0131673, 0.0154528, 0.0131673, 0.0154528, -0.0009510, 0.0009417
9: 0.9183307, 0.9296775, 0.9183307, 0.9296775, -0.0045303, 0.0045702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.ADV_EXAMPLE
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0029602, upper bound: 0.0028075
time: 0.42 seconds

## RS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (rs) = 2.59 + 2.94 = 5.53 seconds
