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
Threshold: 0.0014131413


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127)
1: (-0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604)
2: (0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400)
3: (-0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795)
4: (-0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481)
5: (0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534)
6: (-0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227)
7: (0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557)
8: (-0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554)
9: (-0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 1.39 = 3.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0014214, upper bound: 0.0014214

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.47
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.47
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.47
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.47
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
time: 0.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.55
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.55
Output dim: 7, lower bound: -0.0014157, upper bound: 0.0014157
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.55
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.55
Output dim: 7, lower bound: -0.0014055, upper bound: 0.0014055

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0018279, 0.0006848, -0.0018279, 0.0006848, -0.0025127, 0.0025127
1: -0.0038332, -0.0025728, -0.0038332, -0.0025728, -0.0012604, 0.0012604
2: 0.0317366, 0.0339766, 0.0317366, 0.0339766, -0.0022400, 0.0022400
3: -0.0028670, -0.0005874, -0.0028670, -0.0005874, -0.0022795, 0.0022795
4: -0.0025730, -0.0005249, -0.0025730, -0.0005249, -0.0020481, 0.0020481
5: 0.0108831, 0.0132365, 0.0108831, 0.0132365, -0.0023534, 0.0023534
6: -0.0042366, -0.0017139, -0.0042366, -0.0017139, -0.0025227, 0.0025227
7: 0.9750216, 0.9769773, 0.9750216, 0.9769773, -0.0019557, 0.0019557
8: -0.0133048, -0.0061494, -0.0133048, -0.0061494, -0.0071554, 0.0071554
9: -0.0004434, 0.0036575, -0.0004434, 0.0036575, -0.0041009, 0.0041009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
time: 0.52 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.50
Output dim: 7, lower bound: -0.0014118, upper bound: 0.0014118

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.01 + 36.50 = 39.50 seconds
