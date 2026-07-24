## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13486788544


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476)
1: (-0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610)
2: (-0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824)
3: (-0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275)
4: (-0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858)
5: (-0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087)
6: (-0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672)
7: (0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243)
8: (-0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1799640, 0.1799640)
9: (-0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.86 = 3.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1795345, 0.1795726
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1795727, 0.1795345
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1788207, 0.1789492
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1788971, 0.1788588
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1793831, 0.1792277
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1792659, 0.1793336
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1366649, upper bound: 0.1366649
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1366649, upper bound: 0.1366649
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.1366649, upper bound: 0.1366649
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 7, lower bound: -0.1366649, upper bound: 0.1366649

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1754539, 0.1760906
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1759704, 0.1755824
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378329, upper bound: 0.1378329
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378329, upper bound: 0.1378329
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1755303, 0.1760074
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1760473, 0.1754920
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1335559, upper bound: 0.1335559
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1335559, upper bound: 0.1335559
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1788105, 0.1785986
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373832, upper bound: 0.1373832
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373832, upper bound: 0.1373832
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1787539, 0.1786400
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1791921, 0.1793060
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1792659, 0.1792598
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1366649, upper bound: 0.1366649
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1366649, upper bound: 0.1366649
time: 0.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1378329, upper bound: 0.1378329
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1378329, upper bound: 0.1378329
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1335559, upper bound: 0.1335559
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1335559, upper bound: 0.1335559
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1373832, upper bound: 0.1373832
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1373832, upper bound: 0.1373832
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1366649, upper bound: 0.1366649
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 7, lower bound: -0.1366649, upper bound: 0.1366649

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1751971, 0.1758899
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1752771, 0.1758338
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377415, upper bound: 0.1377415
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377415, upper bound: 0.1377415
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1758492, 0.1754710
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1758589, 0.1754534
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 2.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1734107, 0.1742025
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1737314, 0.1738878
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1781002, 0.1781283
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373753, upper bound: 0.1373753
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373753, upper bound: 0.1373753
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1783500, 0.1778882
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373438, upper bound: 0.1373438
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373438, upper bound: 0.1373438
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1784950, 0.1784372
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1785668, 0.1783811
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1324047, upper bound: 0.1324047
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1324047, upper bound: 0.1324047
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1791624, 0.1792900
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1791761, 0.1792750
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1771063, 0.1774485
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1774507, 0.1770996
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1176502, upper bound: 0.1176502
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1176502, upper bound: 0.1176502
time: 1.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1377415, upper bound: 0.1377415
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1377415, upper bound: 0.1377415
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1373753, upper bound: 0.1373753
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1373753, upper bound: 0.1373753
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1373438, upper bound: 0.1373438
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1373438, upper bound: 0.1373438
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1324047, upper bound: 0.1324047
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1324047, upper bound: 0.1324047
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1176502, upper bound: 0.1176502
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.13
Output dim: 7, lower bound: -0.1176502, upper bound: 0.1176502

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1750223, 0.1757574
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365708, upper bound: 0.1365708
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365708, upper bound: 0.1365708
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1750646, 0.1757148
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1724826, 0.1734734
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377415, upper bound: 0.1377415
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377415, upper bound: 0.1377415
time: 2.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1729498, 0.1730394
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376695, upper bound: 0.1376695
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376695, upper bound: 0.1376695
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1752730, 0.1748542
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377212, upper bound: 0.1377212
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377212, upper bound: 0.1377212
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1752325, 0.1748930
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1752817, 0.1748367
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371777, upper bound: 0.1371777
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371777, upper bound: 0.1371777
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1752422, 0.1748731
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1733663, 0.1741719
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1733802, 0.1741578
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1334211, upper bound: 0.1334211
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1334211, upper bound: 0.1334211
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1731324, 0.1732543
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1730979, 0.1732866
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1331478, upper bound: 0.1331478
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1331478, upper bound: 0.1331478
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1748191, 0.1753504
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1753149, 0.1748472
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372744, upper bound: 0.1372744
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372744, upper bound: 0.1372744
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1780984, 0.1777165
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1781550, 0.1776366
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373438, upper bound: 0.1373438
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373438, upper bound: 0.1373438
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1784637, 0.1784196
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1784773, 0.1784053
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364750, upper bound: 0.1364750
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364750, upper bound: 0.1364750
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1758569, 0.1765147
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365482, upper bound: 0.1365482
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365482, upper bound: 0.1365482
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1763728, 0.1759845
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1785836, 0.1786371
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359962, upper bound: 0.1359962
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359962, upper bound: 0.1359962
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1785381, 0.1787030
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1770652, 0.1774218
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1770789, 0.1774057
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364876, upper bound: 0.1364876
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364876, upper bound: 0.1364876
time: 0.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1365708, upper bound: 0.1365708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1365708, upper bound: 0.1365708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377415, upper bound: 0.1377415
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377415, upper bound: 0.1377415
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1376695, upper bound: 0.1376695
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1376695, upper bound: 0.1376695
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377212, upper bound: 0.1377212
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377212, upper bound: 0.1377212
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1371777, upper bound: 0.1371777
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1371777, upper bound: 0.1371777
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1334211, upper bound: 0.1334211
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1334211, upper bound: 0.1334211
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1331478, upper bound: 0.1331478
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1331478, upper bound: 0.1331478
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1372744, upper bound: 0.1372744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1372744, upper bound: 0.1372744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1373438, upper bound: 0.1373438
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1373438, upper bound: 0.1373438
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1364750, upper bound: 0.1364750
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1364750, upper bound: 0.1364750
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1365482, upper bound: 0.1365482
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1365482, upper bound: 0.1365482
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1359962, upper bound: 0.1359962
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1359962, upper bound: 0.1359962
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1364876, upper bound: 0.1364876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.99
Output dim: 7, lower bound: -0.1364876, upper bound: 0.1364876

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1749579, 0.1757411
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365708, upper bound: 0.1365708
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365708, upper bound: 0.1365708
time: 3.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1750223, 0.1756930
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1748399, 0.1753842
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1747340, 0.1754843
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1723377, 0.1732337
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375987, upper bound: 0.1375987
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375987, upper bound: 0.1375987
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1722429, 0.1733669
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375987, upper bound: 0.1375987
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375987, upper bound: 0.1375987
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1723442, 0.1724057
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1311213, upper bound: 0.1311213
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1311213, upper bound: 0.1311213
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1723160, 0.1724166
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375575, upper bound: 0.1375575
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375575, upper bound: 0.1375575
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1751146, 0.1747217
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1751405, 0.1746808
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369542, upper bound: 0.1369543
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369542, upper bound: 0.1369543
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1730915, 0.1730812
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376591, upper bound: 0.1376591
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376591, upper bound: 0.1376591
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1734065, 0.1727519
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1745753, 0.1743877
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1216086, upper bound: 0.1216086
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1216086, upper bound: 0.1216086
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1747771, 0.1741302
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371240, upper bound: 0.1371240
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371240, upper bound: 0.1371240
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1731012, 0.1730662
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1734147, 0.1727321
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377029, upper bound: 0.1377029
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377029, upper bound: 0.1377029
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727735, 0.1735385
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727330, 0.1736047
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378293, upper bound: 0.1378293
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378293, upper bound: 0.1378293
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1730865, 0.1732238
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373231, upper bound: 0.1373231
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373231, upper bound: 0.1373231
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1731020, 0.1732107
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377583, upper bound: 0.1377583
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377583, upper bound: 0.1377583
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1745199, 0.1751250
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1745716, 0.1750512
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727307, 0.1727670
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1732158, 0.1722630
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1747698, 0.1748838
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1248374, upper bound: 0.1248374
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1248374, upper bound: 0.1248374
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1752355, 0.1743879
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1774194, 0.1769774
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373240, upper bound: 0.1373240
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1373240, upper bound: 0.1373240
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1774786, 0.1769010
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1361011, upper bound: 0.1361011
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1361011, upper bound: 0.1361011
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1763043, 0.1766183
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372926, upper bound: 0.1372926
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372926, upper bound: 0.1372926
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1766415, 0.1762602
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377541, upper bound: 0.1377541
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377541, upper bound: 0.1377541
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1784002, 0.1783692
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1784773, 0.1783281
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1165469, upper bound: 0.1165469
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1165469, upper bound: 0.1165469
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1731823, 0.1743296
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360829, upper bound: 0.1360829
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360829, upper bound: 0.1360829
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1736443, 0.1738402
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364671, upper bound: 0.1364671
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364671, upper bound: 0.1364671
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1758256, 0.1754049
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364514, upper bound: 0.1364514
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364514, upper bound: 0.1364514
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1757932, 0.1754701
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1778763, 0.1781847
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1781138, 0.1779298
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1358047, upper bound: 0.1358047
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1358047, upper bound: 0.1358047
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1763980, 0.1769054
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1362080, upper bound: 0.1362080
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1362080, upper bound: 0.1362080
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1767424, 0.1765628
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1165844, upper bound: 0.1165844
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1165844, upper bound: 0.1165844
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1738372, 0.1747027
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1743532, 0.1741993
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365700, upper bound: 0.1365700
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365700, upper bound: 0.1365700
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1769032, 0.1772714
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359063, upper bound: 0.1359063
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359063, upper bound: 0.1359063
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1769446, 0.1772396
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364876, upper bound: 0.1364876
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364876, upper bound: 0.1364876
time: 0.82 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1365708, upper bound: 0.1365708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1365708, upper bound: 0.1365708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377774, upper bound: 0.1377774
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1375987, upper bound: 0.1375987
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1375987, upper bound: 0.1375987
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1375987, upper bound: 0.1375987
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1375987, upper bound: 0.1375987
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1311213, upper bound: 0.1311213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1311213, upper bound: 0.1311213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1375575, upper bound: 0.1375575
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1375575, upper bound: 0.1375575
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1369542, upper bound: 0.1369543
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1369542, upper bound: 0.1369543
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1376591, upper bound: 0.1376591
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1376591, upper bound: 0.1376591
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1216086, upper bound: 0.1216086
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1216086, upper bound: 0.1216086
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1371240, upper bound: 0.1371240
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1371240, upper bound: 0.1371240
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377852, upper bound: 0.1377852
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377029, upper bound: 0.1377029
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377029, upper bound: 0.1377029
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1378559, upper bound: 0.1378559
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1378293, upper bound: 0.1378293
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1378293, upper bound: 0.1378293
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1373231, upper bound: 0.1373231
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1373231, upper bound: 0.1373231
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377583, upper bound: 0.1377583
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377583, upper bound: 0.1377583
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1248374, upper bound: 0.1248374
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1248374, upper bound: 0.1248374
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1373241, upper bound: 0.1373241
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1373240, upper bound: 0.1373240
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1373240, upper bound: 0.1373240
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1361011, upper bound: 0.1361011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1361011, upper bound: 0.1361011
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1372926, upper bound: 0.1372926
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1372926, upper bound: 0.1372926
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377541, upper bound: 0.1377541
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1377541, upper bound: 0.1377541
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1165469, upper bound: 0.1165469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1165469, upper bound: 0.1165469
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1360829, upper bound: 0.1360829
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1360829, upper bound: 0.1360829
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364671, upper bound: 0.1364671
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364671, upper bound: 0.1364671
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364514, upper bound: 0.1364514
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364514, upper bound: 0.1364514
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1358047, upper bound: 0.1358047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1358047, upper bound: 0.1358047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1362080, upper bound: 0.1362080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1362080, upper bound: 0.1362080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1165844, upper bound: 0.1165844
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1165844, upper bound: 0.1165844
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1365719, upper bound: 0.1365719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1365700, upper bound: 0.1365700
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1365700, upper bound: 0.1365700
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1359063, upper bound: 0.1359063
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1359063, upper bound: 0.1359063
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364876, upper bound: 0.1364876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.98
Output dim: 7, lower bound: -0.1364876, upper bound: 0.1364876

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1747356, 0.1754097
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1746265, 0.1755161
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360070, upper bound: 0.1360071
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360071, upper bound: 0.1360070
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1748960, 0.1755811
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1749104, 0.1755699
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1741769, 0.1749688
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360070, upper bound: 0.1360071
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360070, upper bound: 0.1360071
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1744272, 0.1747212
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1726151, 0.1736393
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377379, upper bound: 0.1377379
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377379, upper bound: 0.1377379
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1729561, 0.1733655
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1722132, 0.1731215
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375487, upper bound: 0.1375487
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375487, upper bound: 0.1375487
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1722256, 0.1731075
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1299407, upper bound: 0.1299407
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1299407, upper bound: 0.1299407
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1721226, 0.1732547
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363994, upper bound: 0.1363994
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363994, upper bound: 0.1363994
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1721307, 0.1732363
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371671, upper bound: 0.1371671
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371671, upper bound: 0.1371671
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1721516, 0.1722868
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364614, upper bound: 0.1364614
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364614, upper bound: 0.1364614
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1721861, 0.1722430
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375575, upper bound: 0.1375575
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1375575, upper bound: 0.1375575
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1723404, 0.1724073
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727928, 0.1719475
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369353, upper bound: 0.1369353
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369353, upper bound: 0.1369353
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1744341, 0.1742325
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369542, upper bound: 0.1369543
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369542, upper bound: 0.1369542
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1746367, 0.1739744
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369353, upper bound: 0.1369353
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369353, upper bound: 0.1369353
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1728092, 0.1728634
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1362842, upper bound: 0.1362842
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1362842, upper bound: 0.1362842
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1728710, 0.1727989
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376591, upper bound: 0.1376591
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376591, upper bound: 0.1376591
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1733347, 0.1727216
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1734065, 0.1726801
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1747386, 0.1741051
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1356201, upper bound: 0.1356201
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1356201, upper bound: 0.1356201
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1747520, 0.1740916
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1729265, 0.1727780
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377583, upper bound: 0.1377583
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377583, upper bound: 0.1377583
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1728129, 0.1728870
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377029, upper bound: 0.1377029
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1377029, upper bound: 0.1377029
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1706210, 0.1704040
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1710756, 0.1699384
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1725984, 0.1732524
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378293, upper bound: 0.1378293
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1378293, upper bound: 0.1378293
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1724874, 0.1733790
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1329330, upper bound: 0.1329330
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1329330, upper bound: 0.1329330
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1724512, 0.1733548
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1317897, upper bound: 0.1317897
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1317897, upper bound: 0.1317897
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1725347, 0.1733229
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1723704, 0.1727377
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369877, upper bound: 0.1369877
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369877, upper bound: 0.1369877
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1725989, 0.1725077
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1729695, 0.1730947
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1729859, 0.1730863
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1743857, 0.1750120
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371325, upper bound: 0.1371325
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371325, upper bound: 0.1371325
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1744070, 0.1750078
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371325, upper bound: 0.1371325
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371325, upper bound: 0.1371325
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1744425, 0.1749383
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1199821, upper bound: 0.1199821
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1199821, upper bound: 0.1199821
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1744586, 0.1749319
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727044, 0.1727538
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1250938, upper bound: 0.1250938
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1250938, upper bound: 0.1250938
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727175, 0.1727408
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359701, upper bound: 0.1359701
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359701, upper bound: 0.1359701
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1731907, 0.1722498
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1370907, upper bound: 0.1370907
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1370907, upper bound: 0.1370907
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1732026, 0.1722364
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1731667, 0.1726596
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1734997, 0.1723191
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360834, upper bound: 0.1360834
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360834, upper bound: 0.1360834
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1740303, 0.1741053
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1248374, upper bound: 0.1248374
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1248374, upper bound: 0.1248374
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1744846, 0.1735883
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1774029, 0.1768684
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360898, upper bound: 0.1360898
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360898, upper bound: 0.1360898
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1774786, 0.1768253
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1357712, upper bound: 0.1357712
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1357712, upper bound: 0.1357712
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1755689, 0.1760936
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1758271, 0.1758829
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371335, upper bound: 0.1371335
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1371335, upper bound: 0.1371335
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1764682, 0.1761263
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363827, upper bound: 0.1363827
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363827, upper bound: 0.1363827
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1765076, 0.1760933
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369857, upper bound: 0.1369857
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1369857, upper bound: 0.1369857
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1750987, 0.1755641
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1756187, 0.1750677
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1361802, upper bound: 0.1361802
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1361802, upper bound: 0.1361802
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1725293, 0.1739749
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1357370, upper bound: 0.1357370
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1357370, upper bound: 0.1357370
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727870, 0.1736767
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360824, upper bound: 0.1360824
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360824, upper bound: 0.1360824
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1734723, 0.1737102
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364645, upper bound: 0.1364645
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364645, upper bound: 0.1364645
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1735144, 0.1736723
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359037, upper bound: 0.1359037
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359037, upper bound: 0.1359037
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1732017, 0.1732905
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1165844, upper bound: 0.1165844
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1165844, upper bound: 0.1165844
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1736559, 0.1727811
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364514, upper bound: 0.1364514
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364514, upper bound: 0.1364514
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1754838, 0.1752085
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1361802, upper bound: 0.1361802
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1361802, upper bound: 0.1361802
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1755436, 0.1751607
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1757061, 0.1763471
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1760577, 0.1760146
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359809, upper bound: 0.1359809
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359809, upper bound: 0.1359809
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1779614, 0.1778017
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1354964, upper bound: 0.1354964
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1354964, upper bound: 0.1354964
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1779857, 0.1777597
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1358047, upper bound: 0.1358047
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1358047, upper bound: 0.1358047
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1762663, 0.1767877
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1361314, upper bound: 0.1361314
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1361314, upper bound: 0.1361314
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1762803, 0.1767725
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1362040, upper bound: 0.1362040
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1362040, upper bound: 0.1362040
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1729943, 0.1738796
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1730849, 0.1738617
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360915, upper bound: 0.1360915
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360915, upper bound: 0.1360915
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1741048, 0.1740196
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365700, upper bound: 0.1365700
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1365700, upper bound: 0.1365700
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1741646, 0.1739507
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364800, upper bound: 0.1364801
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364800, upper bound: 0.1364800
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1761783, 0.1767824
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359016, upper bound: 0.1359016
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359016, upper bound: 0.1359016
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1764449, 0.1765497
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1357974, upper bound: 0.1357974
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1357974, upper bound: 0.1357974
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1737126, 0.1745179
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359014, upper bound: 0.1359014
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359014, upper bound: 0.1359014
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1742280, 0.1740131
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364671, upper bound: 0.1364671
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1364671, upper bound: 0.1364671
time: 0.90 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360070, upper bound: 0.1360071
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360071, upper bound: 0.1360070
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360070, upper bound: 0.1360071
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360070, upper bound: 0.1360071
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1377379, upper bound: 0.1377379
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1377379, upper bound: 0.1377379
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371019, upper bound: 0.1371019
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1375487, upper bound: 0.1375487
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1375487, upper bound: 0.1375487
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1299407, upper bound: 0.1299407
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1299407, upper bound: 0.1299407
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363994, upper bound: 0.1363994
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363994, upper bound: 0.1363994
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371671, upper bound: 0.1371671
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371671, upper bound: 0.1371671
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364614, upper bound: 0.1364614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364614, upper bound: 0.1364614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1375575, upper bound: 0.1375575
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1375575, upper bound: 0.1375575
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369353, upper bound: 0.1369353
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369353, upper bound: 0.1369353
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369542, upper bound: 0.1369543
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369542, upper bound: 0.1369542
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369353, upper bound: 0.1369353
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369353, upper bound: 0.1369353
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1362842, upper bound: 0.1362842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1362842, upper bound: 0.1362842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1376591, upper bound: 0.1376591
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1376591, upper bound: 0.1376591
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1356201, upper bound: 0.1356201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1356201, upper bound: 0.1356201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1377583, upper bound: 0.1377583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1377583, upper bound: 0.1377583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1377029, upper bound: 0.1377029
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1377029, upper bound: 0.1377029
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363088, upper bound: 0.1363088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1376090, upper bound: 0.1376090
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1378293, upper bound: 0.1378293
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1378293, upper bound: 0.1378293
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1329330, upper bound: 0.1329330
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1329330, upper bound: 0.1329330
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1317897, upper bound: 0.1317897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1317897, upper bound: 0.1317897
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369877, upper bound: 0.1369877
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369877, upper bound: 0.1369877
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371239, upper bound: 0.1371239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371325, upper bound: 0.1371325
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371325, upper bound: 0.1371325
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371325, upper bound: 0.1371325
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371325, upper bound: 0.1371325
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1199821, upper bound: 0.1199821
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1199821, upper bound: 0.1199821
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371341, upper bound: 0.1371341
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1250938, upper bound: 0.1250938
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1250938, upper bound: 0.1250938
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359701, upper bound: 0.1359701
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359701, upper bound: 0.1359701
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1370907, upper bound: 0.1370907
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1370907, upper bound: 0.1370907
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372202, upper bound: 0.1372202
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360834, upper bound: 0.1360834
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360834, upper bound: 0.1360834
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1248374, upper bound: 0.1248374
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1248374, upper bound: 0.1248374
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1372726, upper bound: 0.1372726
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360898, upper bound: 0.1360898
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360898, upper bound: 0.1360898
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1357712, upper bound: 0.1357712
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1357712, upper bound: 0.1357712
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371335, upper bound: 0.1371335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1371335, upper bound: 0.1371335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363827, upper bound: 0.1363827
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1363827, upper bound: 0.1363827
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369857, upper bound: 0.1369857
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1369857, upper bound: 0.1369857
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1361802, upper bound: 0.1361802
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1361802, upper bound: 0.1361802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1357370, upper bound: 0.1357370
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1357370, upper bound: 0.1357370
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360824, upper bound: 0.1360824
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360824, upper bound: 0.1360824
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364645, upper bound: 0.1364645
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364645, upper bound: 0.1364645
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359037, upper bound: 0.1359037
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359037, upper bound: 0.1359037
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1165844, upper bound: 0.1165844
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1165844, upper bound: 0.1165844
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364514, upper bound: 0.1364514
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364514, upper bound: 0.1364514
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1361802, upper bound: 0.1361802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1361802, upper bound: 0.1361802
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364738, upper bound: 0.1364738
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359958, upper bound: 0.1359958
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359809, upper bound: 0.1359809
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359809, upper bound: 0.1359809
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1354964, upper bound: 0.1354964
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1354964, upper bound: 0.1354964
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1358047, upper bound: 0.1358047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1358047, upper bound: 0.1358047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1361314, upper bound: 0.1361314
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1361314, upper bound: 0.1361314
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1362040, upper bound: 0.1362040
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1362040, upper bound: 0.1362040
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364779, upper bound: 0.1364779
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360915, upper bound: 0.1360915
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1360915, upper bound: 0.1360915
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1365700, upper bound: 0.1365700
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1365700, upper bound: 0.1365700
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364800, upper bound: 0.1364801
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364800, upper bound: 0.1364800
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359016, upper bound: 0.1359016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359016, upper bound: 0.1359016
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1357974, upper bound: 0.1357974
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1357974, upper bound: 0.1357974
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359014, upper bound: 0.1359014
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1359014, upper bound: 0.1359014
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364671, upper bound: 0.1364671
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.24
Output dim: 7, lower bound: -0.1364671, upper bound: 0.1364671

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1746070, 0.1752969
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1746228, 0.1752855
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1356586, upper bound: 0.1356586
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1356586, upper bound: 0.1356586
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1739649, 0.1751174
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1356586, upper bound: 0.1356586
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1356586, upper bound: 0.1356586
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1742264, 0.1748546
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360070, upper bound: 0.1360071
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1360070, upper bound: 0.1360070
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727760, 0.1737359
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1731306, 0.1734574
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1727904, 0.1737237
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1731374, 0.1734463
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1363307, upper bound: 0.1363307
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1741131, 0.1749565
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359001, upper bound: 0.1359001
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359001, upper bound: 0.1359001
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1741769, 0.1749051
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359001, upper bound: 0.1359001
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1359001, upper bound: 0.1359001
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1722869, 0.1728521
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1370570, upper bound: 0.1370570
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1370570, upper bound: 0.1370570
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0627977, 0.0236499, -0.0627977, 0.0236499, -0.0864476, 0.0864476
1: -0.0579183, 0.0412426, -0.0579183, 0.0412426, -0.0991610, 0.0991610
2: -0.0621646, 0.1311178, -0.0621646, 0.1311178, -0.1932824, 0.1932824
3: -0.0198100, 0.0647175, -0.0198100, 0.0647175, -0.0845275, 0.0845275
4: -0.0592303, 0.0713555, -0.0592303, 0.0713555, -0.1305858, 0.1305858
5: -0.0451864, 0.0562223, -0.0451864, 0.0562223, -0.1014087, 0.1014087
6: -0.1142149, 0.0745523, -0.1142149, 0.0745523, -0.1887672, 0.1887672
7: 0.8356043, 1.0151286, 0.8356043, 1.0151286, -0.1795243, 0.1795243
8: -0.0721872, 0.1097672, -0.0721872, 0.1097672, -0.1726180, 0.1725809
9: -0.0869121, 0.0732940, -0.0869121, 0.0732940, -0.1602061, 0.1602061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.39 + 597.87 = 601.26 seconds
