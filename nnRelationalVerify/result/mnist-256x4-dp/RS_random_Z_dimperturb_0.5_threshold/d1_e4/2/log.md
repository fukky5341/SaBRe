## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00066248


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084)
1: (-0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797)
2: (-0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143)
3: (1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560)
4: (-0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655)
5: (0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274544, 0.0274544)
6: (-0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050)
7: (-0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132206, 0.0132206)
8: (-0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382)
9: (-0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.87 = 3.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0016995, upper bound: 0.0016995

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 154
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 154

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
time: 0.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274513, 0.0274544
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132206, 0.0132201
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 1.72 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 162

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
time: 0.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274544, 0.0274544
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132206, 0.0132206
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162

Time for candidate selection: 1.70 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010000, upper bound: 0.0009980
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009980, upper bound: 0.0010000
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -0.0010000, upper bound: 0.0009980
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 3, lower bound: -0.0009980, upper bound: 0.0010000

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274447, 0.0274486
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132191
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010159, upper bound: 0.0010116
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010113, upper bound: 0.0010159
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274453, 0.0274478
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132196, 0.0132193
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010170
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274490, 0.0274498
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132198
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010000, upper bound: 0.0009980
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009999, upper bound: 0.0009979
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274498, 0.0274490
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132198, 0.0132199
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008851
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008851
time: 0.60 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 3, lower bound: -0.0010159, upper bound: 0.0010116
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 3, lower bound: -0.0010113, upper bound: 0.0010159
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010170
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 3, lower bound: -0.0010000, upper bound: 0.0009980
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 3, lower bound: -0.0009999, upper bound: 0.0009979
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008851
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008851

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274441, 0.0274501
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132190
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009860, upper bound: 0.0009760
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009786, upper bound: 0.0009816
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274467, 0.0274472
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132195, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009810, upper bound: 0.0009775
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009764, upper bound: 0.0009860
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274627, 0.0274648
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132219, 0.0132217
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274619, 0.0274658
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132221, 0.0132215
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010170
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010170
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274141, 0.0274197
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132155, 0.0132146
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009612
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009627, upper bound: 0.0009680
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274197, 0.0274141
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132146, 0.0132155
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008850, upper bound: 0.0008829
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008850, upper bound: 0.0008829
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273447, 0.0273296
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132024, 0.0132046
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008845
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008826, upper bound: 0.0008851
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273296, 0.0273448
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132046, 0.0132024
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008827, upper bound: 0.0008851
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008851
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0009860, upper bound: 0.0009760
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0009786, upper bound: 0.0009816
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0009810, upper bound: 0.0009775
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0009764, upper bound: 0.0009860
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010171
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010170
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0010171, upper bound: 0.0010170
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009612
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0009627, upper bound: 0.0009680
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0008850, upper bound: 0.0008829
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0008850, upper bound: 0.0008829
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008845
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0008826, upper bound: 0.0008851
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0008827, upper bound: 0.0008851
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008851

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274508, 0.0274539
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009689, upper bound: 0.0009581
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009665, upper bound: 0.0009589
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274508, 0.0274538
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009615, upper bound: 0.0009636
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009602, upper bound: 0.0009642
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274508, 0.0274539
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009769, upper bound: 0.0009735
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009717, upper bound: 0.0009735
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274508, 0.0274538
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009763, upper bound: 0.0009860
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009764, upper bound: 0.0009860
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274110, 0.0274197
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132155, 0.0132142
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010130, upper bound: 0.0010071
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010068, upper bound: 0.0010130
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274169, 0.0274141
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132146, 0.0132151
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009021, upper bound: 0.0009022
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009021, upper bound: 0.0009022
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274110, 0.0274197
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132155, 0.0132142
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010159, upper bound: 0.0010113
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010115, upper bound: 0.0010158
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274169, 0.0274141
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132146, 0.0132151
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010133, upper bound: 0.0010164
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010164, upper bound: 0.0010134
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274538, 0.0274539
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132205
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009661, upper bound: 0.0009569
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009593, upper bound: 0.0009572
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274539, 0.0274538
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132205
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009627, upper bound: 0.0009678
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009605, upper bound: 0.0009680
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273447, 0.0273296
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132024, 0.0132046
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008809, upper bound: 0.0008747
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008751, upper bound: 0.0008789
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273296, 0.0273448
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132046, 0.0132024
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008837, upper bound: 0.0008790
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008817
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274487, 0.0274494
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132197
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008794
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008790, upper bound: 0.0008833
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274494, 0.0274487
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132198
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008822, upper bound: 0.0008851
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008826, upper bound: 0.0008851
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274478, 0.0274486
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008827, upper bound: 0.0008845
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008822, upper bound: 0.0008851
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274486, 0.0274478
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132196, 0.0132197
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008795
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008790, upper bound: 0.0008839
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009689, upper bound: 0.0009581
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009665, upper bound: 0.0009589
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009615, upper bound: 0.0009636
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009602, upper bound: 0.0009642
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009769, upper bound: 0.0009735
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009717, upper bound: 0.0009735
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009763, upper bound: 0.0009860
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009764, upper bound: 0.0009860
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0010130, upper bound: 0.0010071
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0010068, upper bound: 0.0010130
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009021, upper bound: 0.0009022
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009021, upper bound: 0.0009022
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0010159, upper bound: 0.0010113
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0010115, upper bound: 0.0010158
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0010133, upper bound: 0.0010164
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0010164, upper bound: 0.0010134
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009661, upper bound: 0.0009569
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009593, upper bound: 0.0009572
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009627, upper bound: 0.0009678
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0009605, upper bound: 0.0009680
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008809, upper bound: 0.0008747
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008751, upper bound: 0.0008789
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008837, upper bound: 0.0008790
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008817
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008794
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008790, upper bound: 0.0008833
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008822, upper bound: 0.0008851
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008826, upper bound: 0.0008851
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008827, upper bound: 0.0008845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008822, upper bound: 0.0008851
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008795
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.35
Output dim: 3, lower bound: -0.0008790, upper bound: 0.0008839

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274459, 0.0274498
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132193
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009689, upper bound: 0.0009581
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009689, upper bound: 0.0009581
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274466, 0.0274490
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132198, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009625, upper bound: 0.0009540
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009575, upper bound: 0.0009549
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274459, 0.0274498
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132193
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009615, upper bound: 0.0009636
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009607, upper bound: 0.0009636
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274466, 0.0274490
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132198, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009562, upper bound: 0.0009548
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009559, upper bound: 0.0009602
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274477, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009769, upper bound: 0.0009716
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009768, upper bound: 0.0009735
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274502, 0.0274512
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132201, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009717, upper bound: 0.0009716
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009715, upper bound: 0.0009735
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274627, 0.0274648
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132219, 0.0132217
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009763, upper bound: 0.0009860
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009761, upper bound: 0.0009860
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274619, 0.0274658
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132221, 0.0132215
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008617, upper bound: 0.0008711
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008618, upper bound: 0.0008711
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274477, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009958, upper bound: 0.0009893
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009939, upper bound: 0.0009897
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274502, 0.0274512
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132201, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 64

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009759, upper bound: 0.0009752
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009744, upper bound: 0.0009832
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273416, 0.0273296
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132024, 0.0132042
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Candidate
type: RSZ, layer: 3, pos: 45

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008849, upper bound: 0.0008826
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008851
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273265, 0.0273447
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132046, 0.0132019
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008981, upper bound: 0.0008924
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008922, upper bound: 0.0008981
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274441, 0.0274501
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132190
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010121, upper bound: 0.0010106
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010152, upper bound: 0.0010076
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274467, 0.0274472
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132195, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009815, upper bound: 0.0009777
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009760, upper bound: 0.0009860
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274471, 0.0274482
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132195
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008983, upper bound: 0.0009014
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008983, upper bound: 0.0009015
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274451, 0.0274504
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132200, 0.0132192
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009992, upper bound: 0.0009954
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009973, upper bound: 0.0009961
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274512, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132201
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009661, upper bound: 0.0009565
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009661, upper bound: 0.0009569
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274533, 0.0274512
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132201, 0.0132204
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009581, upper bound: 0.0009544
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009547, upper bound: 0.0009560
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274487, 0.0274494
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132197
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009615, upper bound: 0.0009634
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009592, upper bound: 0.0009666
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274494, 0.0274487
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132198
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009565, upper bound: 0.0009586
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009561, upper bound: 0.0009640
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274512, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132201
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008710
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008757, upper bound: 0.0008735
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274533, 0.0274512
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132201, 0.0132204
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008743, upper bound: 0.0008782
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008745, upper bound: 0.0008764
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274472, 0.0274501
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132195
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008837, upper bound: 0.0008790
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008837, upper bound: 0.0008785
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274501, 0.0274472
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132195, 0.0132199
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008817
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008796, upper bound: 0.0008817
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274472, 0.0274501
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132195
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008794
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008794
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274501, 0.0274472
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132195, 0.0132199
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008750, upper bound: 0.0008734
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008711, upper bound: 0.0008792
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274478, 0.0274486
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008822, upper bound: 0.0008850
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008822, upper bound: 0.0008851
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274486, 0.0274478
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132196, 0.0132197
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008814, upper bound: 0.0008795
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008788, upper bound: 0.0008839
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274487, 0.0274494
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132197
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008527, upper bound: 0.0008452
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008466, upper bound: 0.0008547
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274494, 0.0274487
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132198
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008845
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008816, upper bound: 0.0008813
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274472, 0.0274501
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132195
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008795
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008795
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274501, 0.0274472
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132195, 0.0132199
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008790, upper bound: 0.0008839
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008789, upper bound: 0.0008837
time: 0.62 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009689, upper bound: 0.0009581
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009689, upper bound: 0.0009581
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009625, upper bound: 0.0009540
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009575, upper bound: 0.0009549
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009615, upper bound: 0.0009636
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009607, upper bound: 0.0009636
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009562, upper bound: 0.0009548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009559, upper bound: 0.0009602
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009769, upper bound: 0.0009716
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009768, upper bound: 0.0009735
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009717, upper bound: 0.0009716
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009715, upper bound: 0.0009735
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009763, upper bound: 0.0009860
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009761, upper bound: 0.0009860
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008617, upper bound: 0.0008711
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008618, upper bound: 0.0008711
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009958, upper bound: 0.0009893
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009939, upper bound: 0.0009897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009759, upper bound: 0.0009752
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009744, upper bound: 0.0009832
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008849, upper bound: 0.0008826
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0008851
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008981, upper bound: 0.0008924
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008922, upper bound: 0.0008981
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0010121, upper bound: 0.0010106
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0010152, upper bound: 0.0010076
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009815, upper bound: 0.0009777
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009760, upper bound: 0.0009860
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008983, upper bound: 0.0009014
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008983, upper bound: 0.0009015
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009992, upper bound: 0.0009954
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009973, upper bound: 0.0009961
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009661, upper bound: 0.0009565
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009661, upper bound: 0.0009569
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009581, upper bound: 0.0009544
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009547, upper bound: 0.0009560
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009615, upper bound: 0.0009634
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009592, upper bound: 0.0009666
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009565, upper bound: 0.0009586
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0009561, upper bound: 0.0009640
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008710
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008757, upper bound: 0.0008735
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008743, upper bound: 0.0008782
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008745, upper bound: 0.0008764
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008837, upper bound: 0.0008790
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008837, upper bound: 0.0008785
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008817
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008796, upper bound: 0.0008817
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008794
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008794
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008750, upper bound: 0.0008734
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008711, upper bound: 0.0008792
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008822, upper bound: 0.0008850
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008822, upper bound: 0.0008851
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008814, upper bound: 0.0008795
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008788, upper bound: 0.0008839
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008527, upper bound: 0.0008452
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008466, upper bound: 0.0008547
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008816, upper bound: 0.0008813
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008795
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008795
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008790, upper bound: 0.0008839
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.32
Output dim: 3, lower bound: -0.0008789, upper bound: 0.0008837

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274627, 0.0274648
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132219, 0.0132217
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009648, upper bound: 0.0009538
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009574, upper bound: 0.0009541
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274619, 0.0274658
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132221, 0.0132215
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009689, upper bound: 0.0009581
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009689, upper bound: 0.0009581
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274477, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008474, upper bound: 0.0008395
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008474, upper bound: 0.0008395
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274502, 0.0274512
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132201, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009564, upper bound: 0.0009542
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009568, upper bound: 0.0009532
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274110, 0.0274197
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132155, 0.0132142
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008463, upper bound: 0.0008489
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008466, upper bound: 0.0008490
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274169, 0.0274141
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132146, 0.0132151
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009568, upper bound: 0.0009547
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009564, upper bound: 0.0009595
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274477, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009562, upper bound: 0.0009548
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009560, upper bound: 0.0009547
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274502, 0.0274512
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132201, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009559, upper bound: 0.0009599
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009548, upper bound: 0.0009602
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274456, 0.0274494
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132198, 0.0132193
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009769, upper bound: 0.0009716
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009769, upper bound: 0.0009716
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274463, 0.0274487
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009768, upper bound: 0.0009734
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009768, upper bound: 0.0009735
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274456, 0.0274494
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132198, 0.0132193
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009717, upper bound: 0.0009716
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009717, upper bound: 0.0009711
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274463, 0.0274487
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009715, upper bound: 0.0009734
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009715, upper bound: 0.0009735
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274110, 0.0274197
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132155, 0.0132142
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009591, upper bound: 0.0009668
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009585, upper bound: 0.0009689
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274169, 0.0274141
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132146, 0.0132151
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009589, upper bound: 0.0009667
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009584, upper bound: 0.0009689
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273416, 0.0273296
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132024, 0.0132042
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008617, upper bound: 0.0008711
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008616, upper bound: 0.0008711
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273265, 0.0273447
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132046, 0.0132019
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008578, upper bound: 0.0008601
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008569, upper bound: 0.0008670
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274459, 0.0274498
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132193
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 45

### Candidate
type: RSZ, layer: 3, pos: 254

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009921, upper bound: 0.0009886
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009952, upper bound: 0.0009883
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274466, 0.0274490
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132198, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009639, upper bound: 0.0009577
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009564, upper bound: 0.0009590
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274508, 0.0274539
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 254

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009752, upper bound: 0.0009746
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009753, upper bound: 0.0009740
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274508, 0.0274538
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009572, upper bound: 0.0009636
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009565, upper bound: 0.0009660
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274459, 0.0274498
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132193
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008812, upper bound: 0.0008820
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008843, upper bound: 0.0008802
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274466, 0.0274490
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132198, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008793
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008790, upper bound: 0.0008839
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274477, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008945, upper bound: 0.0008917
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008974, upper bound: 0.0008917
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274502, 0.0274512
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132201, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 64

### Candidate
type: RSZ, layer: 3, pos: 45

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008748, upper bound: 0.0008786
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008743, upper bound: 0.0008810
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274471, 0.0274482
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132195
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010080, upper bound: 0.0010024
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0010053, upper bound: 0.0010066
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274451, 0.0274504
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132200, 0.0132192
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009003, upper bound: 0.0008933
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009003, upper bound: 0.0008932
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274508, 0.0274539
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 162

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009641, upper bound: 0.0009597
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009634, upper bound: 0.0009605
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274508, 0.0274538
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132205, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009720, upper bound: 0.0009747
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009714, upper bound: 0.0009820
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273416, 0.0273296
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132024, 0.0132042
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008983, upper bound: 0.0009007
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008977, upper bound: 0.0009014
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0273265, 0.0273447
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132046, 0.0132019
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008943, upper bound: 0.0008915
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008918, upper bound: 0.0008974
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274459, 0.0274498
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132193
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009992, upper bound: 0.0009944
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009988, upper bound: 0.0009954
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274466, 0.0274490
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132198, 0.0132194
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008823, upper bound: 0.0008813
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008823, upper bound: 0.0008813
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274478, 0.0274486
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009660, upper bound: 0.0009565
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009661, upper bound: 0.0009565
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274486, 0.0274478
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132196, 0.0132197
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008512, upper bound: 0.0008422
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008512, upper bound: 0.0008421
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274472, 0.0274501
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132195
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 237

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009580, upper bound: 0.0009541
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009581, upper bound: 0.0009544
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274501, 0.0274472
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132195, 0.0132199
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 254

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008402, upper bound: 0.0008412
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008402, upper bound: 0.0008411
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274472, 0.0274501
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132195
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009610, upper bound: 0.0009634
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009615, upper bound: 0.0009634
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274501, 0.0274472
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132195, 0.0132199
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 162

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009592, upper bound: 0.0009666
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009589, upper bound: 0.0009659
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274512, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132201
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Candidate
type: RSZ, layer: 3, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009552, upper bound: 0.0009547
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009544, upper bound: 0.0009574
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274533, 0.0274512
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132201, 0.0132204
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009554, upper bound: 0.0009640
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009561, upper bound: 0.0009639
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274472, 0.0274501
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132199, 0.0132195
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008710
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008709
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274501, 0.0274472
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132195, 0.0132199
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Candidate
type: RSZ, layer: 3, pos: 254

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008456, upper bound: 0.0008410
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008403, upper bound: 0.0008429
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274504, 0.0274482
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132200
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008435, upper bound: 0.0008420
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008418, upper bound: 0.0008483
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274482, 0.0274504
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132200, 0.0132197
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008741, upper bound: 0.0008762
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008745, upper bound: 0.0008764
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274478, 0.0274486
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132197, 0.0132196
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008539, upper bound: 0.0008435
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008460, upper bound: 0.0008489
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274486, 0.0274478
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132196, 0.0132197
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008539, upper bound: 0.0008438
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008455, upper bound: 0.0008482
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274658, 0.0274647
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132220, 0.0132221
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 107

### Candidate
type: RSZ, layer: 3, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008797, upper bound: 0.0008814
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008794, upper bound: 0.0008817
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274648, 0.0274658
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132221, 0.0132219
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008746, upper bound: 0.0008810
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008789, upper bound: 0.0008792
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274658, 0.0274647
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132220, 0.0132221
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 73

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008792, upper bound: 0.0008787
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008743
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274648, 0.0274658
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132221, 0.0132219
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008794
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008817, upper bound: 0.0008794
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032755, 0.0152329, -0.0032755, 0.0152329, -0.0185084, 0.0185084
1: -0.0062739, 0.0053057, -0.0062739, 0.0053057, -0.0115797, 0.0115797
2: -0.0033420, 0.0172722, -0.0033420, 0.0172722, -0.0206143, 0.0206143
3: 1.0049368, 1.0071929, 1.0049368, 1.0071929, -0.0022560, 0.0022560
4: -0.0044438, 0.0064217, -0.0044438, 0.0064217, -0.0108655, 0.0108655
5: 0.0034489, 0.0314393, 0.0034489, 0.0314393, -0.0274512, 0.0274533
6: -0.0212261, -0.0025211, -0.0212261, -0.0025211, -0.0187050, 0.0187050
7: -0.0233426, -0.0100454, -0.0233426, -0.0100454, -0.0132204, 0.0132201
8: -0.0159080, -0.0039698, -0.0159080, -0.0039698, -0.0119382, 0.0119382
9: -0.0097585, 0.0111727, -0.0097585, 0.0111727, -0.0209312, 0.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.21 + 598.55 = 601.76 seconds
