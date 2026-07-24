## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01246608


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986)
1: (-0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912)
2: (0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394)
3: (-0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711)
4: (-0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0039527, 0.0039527)
5: (-0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973)
6: (-0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037)
7: (-0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0221395, 0.0221395)
8: (0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723)
9: (-0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0153951, 0.0153951)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 2.41 = 3.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0138512, upper bound: 0.0138512

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135602, upper bound: 0.0136414
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136414, upper bound: 0.0135602
time: 1.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.09
Output dim: 8, lower bound: -0.0135602, upper bound: 0.0136414
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.09
Output dim: 8, lower bound: -0.0136414, upper bound: 0.0135602

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0039423, 0.0039429
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0221087, 0.0221096
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0153700, 0.0153685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130984, upper bound: 0.0133864
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132864, upper bound: 0.0131458
time: 1.08 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0039429, 0.0039423
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0221096, 0.0221087
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0153685, 0.0153700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131458, upper bound: 0.0132864
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133864, upper bound: 0.0130984
time: 1.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 8, lower bound: -0.0130984, upper bound: 0.0133864
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 8, lower bound: -0.0132864, upper bound: 0.0131458
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 8, lower bound: -0.0131458, upper bound: 0.0132864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 8, lower bound: -0.0133864, upper bound: 0.0130984

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037964, 0.0038136
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0211600, 0.0212783
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149358, 0.0148773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130438, upper bound: 0.0133324
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130438, upper bound: 0.0133324
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038129, 0.0037970
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212761, 0.0211609
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148788, 0.0149338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130536, upper bound: 0.0128051
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129470, upper bound: 0.0129026
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037970, 0.0038129
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0211609, 0.0212761
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149338, 0.0148788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130918, upper bound: 0.0132310
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130919, upper bound: 0.0132310
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038136, 0.0037964
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212783, 0.0211600
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148773, 0.0149358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133741, upper bound: 0.0130862
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133698, upper bound: 0.0130865
time: 1.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 8, lower bound: -0.0130438, upper bound: 0.0133324
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 8, lower bound: -0.0130438, upper bound: 0.0133324
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 8, lower bound: -0.0130536, upper bound: 0.0128051
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 8, lower bound: -0.0129470, upper bound: 0.0129026
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 8, lower bound: -0.0130918, upper bound: 0.0132310
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 8, lower bound: -0.0130919, upper bound: 0.0132310
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 8, lower bound: -0.0133741, upper bound: 0.0130862
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 8, lower bound: -0.0133698, upper bound: 0.0130865

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037924, 0.0038098
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210975, 0.0212034
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149311, 0.0148724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130320, upper bound: 0.0133155
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130316, upper bound: 0.0133201
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037924, 0.0038096
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210851, 0.0212125
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149309, 0.0148723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128040, upper bound: 0.0131138
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128093, upper bound: 0.0130946
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038102, 0.0037918
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212760, 0.0211608
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148636, 0.0149262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128151, upper bound: 0.0125863
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128257, upper bound: 0.0125692
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038129, 0.0037943
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212760, 0.0211609
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148711, 0.0149338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128822, upper bound: 0.0128318
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128822, upper bound: 0.0128318
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037930, 0.0038090
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210975, 0.0212012
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149291, 0.0148739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128318, upper bound: 0.0128822
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127418, upper bound: 0.0129802
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037931, 0.0038089
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210860, 0.0212111
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149290, 0.0148738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125950, upper bound: 0.0127187
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125950, upper bound: 0.0127187
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038135, 0.0037960
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212748, 0.0211525
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148761, 0.0149358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131360, upper bound: 0.0128503
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131561, upper bound: 0.0128450
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038130, 0.0037963
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212708, 0.0211565
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148773, 0.0149338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131296, upper bound: 0.0128508
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131531, upper bound: 0.0128452
time: 1.11 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0130320, upper bound: 0.0133155
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0130316, upper bound: 0.0133201
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0128040, upper bound: 0.0131138
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0128093, upper bound: 0.0130946
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0128151, upper bound: 0.0125863
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0128257, upper bound: 0.0125692
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0128822, upper bound: 0.0128318
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0128822, upper bound: 0.0128318
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0128318, upper bound: 0.0128822
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0127418, upper bound: 0.0129802
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0125950, upper bound: 0.0127187
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0125950, upper bound: 0.0127187
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0131360, upper bound: 0.0128503
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0131561, upper bound: 0.0128450
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0131296, upper bound: 0.0128508
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0131531, upper bound: 0.0128452

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037923, 0.0038093
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210939, 0.0211958
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149292, 0.0148725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127050, upper bound: 0.0131753
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128948, upper bound: 0.0130772
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037920, 0.0038097
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210900, 0.0211998
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149311, 0.0148713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127921, upper bound: 0.0131017
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127971, upper bound: 0.0130824
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036992, 0.0037220
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206112, 0.0207528
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146556, 0.0145795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127923, upper bound: 0.0130981
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127921, upper bound: 0.0131017
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037008, 0.0037164
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206102, 0.0207385
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146382, 0.0145843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127976, upper bound: 0.0130756
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127971, upper bound: 0.0130824
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037171, 0.0037007
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208020, 0.0206854
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145770, 0.0146337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128026, upper bound: 0.0125742
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128004, upper bound: 0.0125725
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037224, 0.0036987
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208171, 0.0206868
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145711, 0.0146503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127545, upper bound: 0.0125048
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127545, upper bound: 0.0125051
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038089, 0.0037904
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212111, 0.0210860
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148661, 0.0149290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128702, upper bound: 0.0128193
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128702, upper bound: 0.0128192
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038090, 0.0037903
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212012, 0.0210975
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148663, 0.0149291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126271, upper bound: 0.0126998
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127455, upper bound: 0.0125228
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037903, 0.0038036
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210975, 0.0212012
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149134, 0.0148663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128247, upper bound: 0.0128759
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128143, upper bound: 0.0128764
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037930, 0.0038063
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210975, 0.0212012
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149214, 0.0148739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127353, upper bound: 0.0129698
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127295, upper bound: 0.0129741
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037912, 0.0038070
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210641, 0.0211792
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149213, 0.0148660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125859, upper bound: 0.0127078
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125779, upper bound: 0.0127085
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037912, 0.0038089
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210541, 0.0212111
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149290, 0.0148661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125859, upper bound: 0.0127078
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125779, upper bound: 0.0127085
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037210, 0.0037052
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208039, 0.0206821
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145903, 0.0146448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125919, upper bound: 0.0123728
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125919, upper bound: 0.0123728
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037266, 0.0037035
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208189, 0.0206816
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145851, 0.0146622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131017, upper bound: 0.0127921
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131017, upper bound: 0.0127921
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037205, 0.0037054
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207999, 0.0206845
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145912, 0.0146428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128734, upper bound: 0.0125429
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127552, upper bound: 0.0126198
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037263, 0.0037038
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208169, 0.0206855
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145863, 0.0146609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131471, upper bound: 0.0128252
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131469, upper bound: 0.0128403
time: 1.21 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127050, upper bound: 0.0131753
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0128948, upper bound: 0.0130772
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127921, upper bound: 0.0131017
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127971, upper bound: 0.0130824
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127923, upper bound: 0.0130981
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127921, upper bound: 0.0131017
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127976, upper bound: 0.0130756
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127971, upper bound: 0.0130824
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0128026, upper bound: 0.0125742
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0128004, upper bound: 0.0125725
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127545, upper bound: 0.0125048
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127545, upper bound: 0.0125051
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0128702, upper bound: 0.0128193
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0128702, upper bound: 0.0128192
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0126271, upper bound: 0.0126998
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127455, upper bound: 0.0125228
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0128247, upper bound: 0.0128759
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0128143, upper bound: 0.0128764
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127353, upper bound: 0.0129698
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127295, upper bound: 0.0129741
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0125859, upper bound: 0.0127078
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0125779, upper bound: 0.0127085
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0125859, upper bound: 0.0127078
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0125779, upper bound: 0.0127085
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0125919, upper bound: 0.0123728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0125919, upper bound: 0.0123728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0131017, upper bound: 0.0127921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0131017, upper bound: 0.0127921
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0128734, upper bound: 0.0125429
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0127552, upper bound: 0.0126198
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0131471, upper bound: 0.0128252
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 8, lower bound: -0.0131469, upper bound: 0.0128403

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037428, 0.0037630
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0209712, 0.0210765
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147898, 0.0147232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126993, upper bound: 0.0131671
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126874, upper bound: 0.0131694
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037471, 0.0037597
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0209796, 0.0210731
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147799, 0.0147365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126537, upper bound: 0.0128690
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126589, upper bound: 0.0128334
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036995, 0.0037230
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206192, 0.0207441
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146576, 0.0145803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127873, upper bound: 0.0130959
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127713, upper bound: 0.0130961
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037012, 0.0037172
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206182, 0.0207290
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146401, 0.0145854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124801, upper bound: 0.0129425
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126585, upper bound: 0.0128343
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036999, 0.0037223
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206107, 0.0207505
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146561, 0.0145813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123100, upper bound: 0.0125419
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123100, upper bound: 0.0125419
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036995, 0.0037227
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206067, 0.0207523
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146574, 0.0145801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123070, upper bound: 0.0125468
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123070, upper bound: 0.0125468
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037014, 0.0037165
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206097, 0.0207332
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146380, 0.0145860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123176, upper bound: 0.0125212
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123176, upper bound: 0.0125212
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037012, 0.0037170
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206073, 0.0207380
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146399, 0.0145852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127921, upper bound: 0.0130748
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127765, upper bound: 0.0130766
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037178, 0.0037011
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208016, 0.0206827
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145779, 0.0146355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0106613, upper bound: 0.0106771
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0106613, upper bound: 0.0106771
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037172, 0.0037013
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207973, 0.0206851
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145788, 0.0146336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127296, upper bound: 0.0125073
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127296, upper bound: 0.0125079
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037184, 0.0036947
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207508, 0.0206121
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145661, 0.0146454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127423, upper bound: 0.0124930
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127415, upper bound: 0.0124920
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037187, 0.0036946
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207424, 0.0206236
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145662, 0.0146456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125148, upper bound: 0.0123679
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126198, upper bound: 0.0122068
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038088, 0.0037899
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212074, 0.0210785
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148648, 0.0149290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128643, upper bound: 0.0128015
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128639, upper bound: 0.0128122
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038083, 0.0037904
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0212023, 0.0210824
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148662, 0.0149271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126153, upper bound: 0.0126868
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127336, upper bound: 0.0125075
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037597, 0.0037452
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210783, 0.0209837
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147308, 0.0147806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126109, upper bound: 0.0126868
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126153, upper bound: 0.0126868
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037631, 0.0037411
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210815, 0.0209746
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147179, 0.0147907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127336, upper bound: 0.0125093
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127336, upper bound: 0.0125075
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037755, 0.0037846
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210471, 0.0211307
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148564, 0.0148228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128121, upper bound: 0.0128638
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128122, upper bound: 0.0128639
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037700, 0.0037888
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210177, 0.0211508
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148699, 0.0148049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128017, upper bound: 0.0128643
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128015, upper bound: 0.0128643
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037781, 0.0037873
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210471, 0.0211307
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148642, 0.0148303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124993, upper bound: 0.0127458
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125153, upper bound: 0.0127350
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037726, 0.0037915
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210177, 0.0211509
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148780, 0.0148124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124317, upper bound: 0.0128399
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125923, upper bound: 0.0127170
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037765, 0.0037879
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210145, 0.0211084
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148642, 0.0148228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122601, upper bound: 0.0125643
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124440, upper bound: 0.0124266
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037711, 0.0037923
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0209884, 0.0211296
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148781, 0.0148052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123320, upper bound: 0.0124736
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123446, upper bound: 0.0124588
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037765, 0.0037897
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210046, 0.0211396
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148714, 0.0148229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123396, upper bound: 0.0124735
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123514, upper bound: 0.0124586
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037710, 0.0037940
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0209766, 0.0211607
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148854, 0.0148051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123320, upper bound: 0.0124736
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123446, upper bound: 0.0124588
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037193, 0.0037034
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207800, 0.0206501
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145828, 0.0146378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125804, upper bound: 0.0123541
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125796, upper bound: 0.0123639
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037192, 0.0037052
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207718, 0.0206821
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145903, 0.0146373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123219, upper bound: 0.0122310
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124468, upper bound: 0.0120469
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037227, 0.0036995
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207523, 0.0206067
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145801, 0.0146574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130961, upper bound: 0.0127713
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130959, upper bound: 0.0127873
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037230, 0.0036995
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207441, 0.0206192
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145803, 0.0146576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128687, upper bound: 0.0126535
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129610, upper bound: 0.0124646
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037179, 0.0037006
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207999, 0.0206845
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145772, 0.0146354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128674, upper bound: 0.0125325
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128650, upper bound: 0.0125388
time: 1.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037205, 0.0037029
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207999, 0.0206845
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145838, 0.0146428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125070, upper bound: 0.0124864
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126175, upper bound: 0.0123136
time: 1.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037134, 0.0036854
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207693, 0.0206104
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145287, 0.0146212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129171, upper bound: 0.0126855
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130059, upper bound: 0.0124987
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037107, 0.0036909
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207521, 0.0206379
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145466, 0.0146118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130917, upper bound: 0.0127874
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130917, upper bound: 0.0127874
time: 1.24 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126993, upper bound: 0.0131671
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126874, upper bound: 0.0131694
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126537, upper bound: 0.0128690
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126589, upper bound: 0.0128334
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127873, upper bound: 0.0130959
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127713, upper bound: 0.0130961
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0124801, upper bound: 0.0129425
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126585, upper bound: 0.0128343
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123100, upper bound: 0.0125419
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123100, upper bound: 0.0125419
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123070, upper bound: 0.0125468
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123070, upper bound: 0.0125468
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123176, upper bound: 0.0125212
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123176, upper bound: 0.0125212
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127921, upper bound: 0.0130748
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127765, upper bound: 0.0130766
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0106613, upper bound: 0.0106771
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0106613, upper bound: 0.0106771
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127296, upper bound: 0.0125073
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127296, upper bound: 0.0125079
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127423, upper bound: 0.0124930
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127415, upper bound: 0.0124920
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0125148, upper bound: 0.0123679
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126198, upper bound: 0.0122068
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128643, upper bound: 0.0128015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128639, upper bound: 0.0128122
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126153, upper bound: 0.0126868
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127336, upper bound: 0.0125075
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126109, upper bound: 0.0126868
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126153, upper bound: 0.0126868
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127336, upper bound: 0.0125093
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0127336, upper bound: 0.0125075
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128121, upper bound: 0.0128638
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128122, upper bound: 0.0128639
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128017, upper bound: 0.0128643
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128015, upper bound: 0.0128643
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0124993, upper bound: 0.0127458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0125153, upper bound: 0.0127350
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0124317, upper bound: 0.0128399
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0125923, upper bound: 0.0127170
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0122601, upper bound: 0.0125643
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0124440, upper bound: 0.0124266
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123320, upper bound: 0.0124736
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123446, upper bound: 0.0124588
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123396, upper bound: 0.0124735
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123514, upper bound: 0.0124586
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123320, upper bound: 0.0124736
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123446, upper bound: 0.0124588
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0125804, upper bound: 0.0123541
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0125796, upper bound: 0.0123639
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0123219, upper bound: 0.0122310
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0124468, upper bound: 0.0120469
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130961, upper bound: 0.0127713
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130959, upper bound: 0.0127873
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128687, upper bound: 0.0126535
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129610, upper bound: 0.0124646
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128674, upper bound: 0.0125325
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0128650, upper bound: 0.0125388
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0125070, upper bound: 0.0124864
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0126175, upper bound: 0.0123136
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0129171, upper bound: 0.0126855
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130059, upper bound: 0.0124987
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130917, upper bound: 0.0127874
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.04
Output dim: 8, lower bound: -0.0130917, upper bound: 0.0127874

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037261, 0.0037420
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0209212, 0.0210054
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147233, 0.0146717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124544, upper bound: 0.0127880
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123813, upper bound: 0.0128905
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037206, 0.0037463
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208910, 0.0210265
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147383, 0.0146538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124442, upper bound: 0.0127880
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123779, upper bound: 0.0128917
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036518, 0.0036703
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205080, 0.0206185
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144989, 0.0144373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126490, upper bound: 0.0128633
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126344, upper bound: 0.0128631
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036532, 0.0036644
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205030, 0.0206015
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144807, 0.0144405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126542, upper bound: 0.0128275
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126408, upper bound: 0.0128276
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036866, 0.0037073
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205715, 0.0206794
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146082, 0.0145406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122954, upper bound: 0.0125337
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122954, upper bound: 0.0125337
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036810, 0.0037101
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205419, 0.0206964
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146179, 0.0145223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125249, upper bound: 0.0127182
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124522, upper bound: 0.0128189
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036489, 0.0036682
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204947, 0.0206084
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144924, 0.0144280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124752, upper bound: 0.0129345
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124634, upper bound: 0.0129365
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036531, 0.0036649
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205015, 0.0206055
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144826, 0.0144402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124194, upper bound: 0.0124388
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123409, upper bound: 0.0125592
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036981, 0.0037205
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205893, 0.0207184
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146486, 0.0145739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119718, upper bound: 0.0123961
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121700, upper bound: 0.0122877
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036981, 0.0037223
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205786, 0.0207505
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146561, 0.0145738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119718, upper bound: 0.0123961
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121700, upper bound: 0.0122877
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036977, 0.0037209
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205850, 0.0207201
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146499, 0.0145725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119720, upper bound: 0.0124007
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121672, upper bound: 0.0122881
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036977, 0.0037227
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205746, 0.0207523
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146574, 0.0145726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119720, upper bound: 0.0124007
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121672, upper bound: 0.0122881
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037000, 0.0037147
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205885, 0.0207011
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146305, 0.0145807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123063, upper bound: 0.0125068
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122968, upper bound: 0.0125069
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036996, 0.0037165
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205776, 0.0207332
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146380, 0.0145786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119929, upper bound: 0.0123787
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121759, upper bound: 0.0122560
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036883, 0.0036999
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205596, 0.0206695
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145863, 0.0145455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124746, upper bound: 0.0129345
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126537, upper bound: 0.0128281
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036844, 0.0037041
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205356, 0.0206904
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146003, 0.0145324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122953, upper bound: 0.0125132
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122953, upper bound: 0.0125132
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037132, 0.0036973
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207315, 0.0206102
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145738, 0.0146287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124731, upper bound: 0.0123709
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125952, upper bound: 0.0122124
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037133, 0.0036973
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207225, 0.0206213
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145740, 0.0146288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127237, upper bound: 0.0124989
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127183, upper bound: 0.0125021
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037191, 0.0036949
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207503, 0.0206077
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145666, 0.0146473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127364, upper bound: 0.0124808
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127334, upper bound: 0.0124871
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037186, 0.0036953
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207483, 0.0206116
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145679, 0.0146459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125025, upper bound: 0.0123543
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126068, upper bound: 0.0121855
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036666, 0.0036469
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206194, 0.0205096
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144230, 0.0144892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125087, upper bound: 0.0123560
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0123623
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036677, 0.0036425
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206214, 0.0205005
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144098, 0.0144913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126138, upper bound: 0.0121989
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126111, upper bound: 0.0122010
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037940, 0.0037697
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0211577, 0.0210005
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148033, 0.0148851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126039, upper bound: 0.0126681
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127277, upper bound: 0.0124942
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037896, 0.0037752
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0211365, 0.0210288
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148210, 0.0148711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126201, upper bound: 0.0125891
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126484, upper bound: 0.0125790
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037587, 0.0037450
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210796, 0.0209688
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147302, 0.0147779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126084, upper bound: 0.0126685
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126082, upper bound: 0.0126795
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037623, 0.0037408
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210830, 0.0209597
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147171, 0.0147882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127276, upper bound: 0.0124916
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127269, upper bound: 0.0124994
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037594, 0.0037446
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210749, 0.0209765
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147289, 0.0147799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123680, upper bound: 0.0124620
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124019, upper bound: 0.0124530
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037589, 0.0037449
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210706, 0.0209802
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147301, 0.0147780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126084, upper bound: 0.0126685
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126082, upper bound: 0.0126795
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037628, 0.0037403
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210780, 0.0209674
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147159, 0.0147899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127277, upper bound: 0.0124942
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127270, upper bound: 0.0125012
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037623, 0.0037408
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210736, 0.0209712
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147173, 0.0147881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124895, upper bound: 0.0123010
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125166, upper bound: 0.0122801
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037755, 0.0037840
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210441, 0.0211228
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148542, 0.0148226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124994, upper bound: 0.0127269
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126795, upper bound: 0.0126082
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037751, 0.0037846
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210403, 0.0211277
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148561, 0.0148212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125012, upper bound: 0.0127270
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126795, upper bound: 0.0126038
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037700, 0.0037883
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210147, 0.0211436
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148680, 0.0148046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124916, upper bound: 0.0127276
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126685, upper bound: 0.0126084
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037696, 0.0037888
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0210114, 0.0211478
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0148697, 0.0148033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124942, upper bound: 0.0127277
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126681, upper bound: 0.0126039
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036869, 0.0037030
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205763, 0.0206771
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145967, 0.0145415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124865, upper bound: 0.0127323
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124874, upper bound: 0.0127334
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036885, 0.0036961
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205746, 0.0206598
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145756, 0.0145463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125021, upper bound: 0.0127183
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125035, upper bound: 0.0127225
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037213, 0.0037437
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208942, 0.0210306
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147319, 0.0146562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121989, upper bound: 0.0126138
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122267, upper bound: 0.0126050
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037261, 0.0037403
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0209047, 0.0210274
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147219, 0.0146704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125797, upper bound: 0.0127038
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125806, upper bound: 0.0127036
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037251, 0.0037404
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208905, 0.0209888
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147181, 0.0146665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120188, upper bound: 0.0123318
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0123181
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036798, 0.0037063
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205174, 0.0206723
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146058, 0.0145161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120113, upper bound: 0.0123321
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121929, upper bound: 0.0122064
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036852, 0.0037054
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205336, 0.0206857
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146038, 0.0145339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123306, upper bound: 0.0124623
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123265, upper bound: 0.0124644
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036798, 0.0037081
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205056, 0.0207035
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146132, 0.0145160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120113, upper bound: 0.0123321
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121929, upper bound: 0.0122064
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037064, 0.0036865
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207330, 0.0205790
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145297, 0.0145982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125132, upper bound: 0.0122953
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125132, upper bound: 0.0122953
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037023, 0.0036905
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207108, 0.0206030
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145431, 0.0145845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125129, upper bound: 0.0123049
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125129, upper bound: 0.0123049
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037098, 0.0036810
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207046, 0.0205310
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145223, 0.0146177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128189, upper bound: 0.0124522
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127182, upper bound: 0.0125250
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037071, 0.0036866
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206876, 0.0205591
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145404, 0.0146081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0122954
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0122954
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036707, 0.0036516
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206205, 0.0205043
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144360, 0.0145001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122881, upper bound: 0.0121672
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122881, upper bound: 0.0121672
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036713, 0.0036472
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206218, 0.0204956
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144228, 0.0145012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129552, upper bound: 0.0124474
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129549, upper bound: 0.0124598
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037051, 0.0036839
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207523, 0.0206133
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145246, 0.0145958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126250, upper bound: 0.0123940
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127317, upper bound: 0.0122266
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037007, 0.0036878
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207317, 0.0206369
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145376, 0.0145817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126245, upper bound: 0.0124006
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127291, upper bound: 0.0122302
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036680, 0.0036545
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206762, 0.0205670
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144381, 0.0144853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124423, upper bound: 0.0124195
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124423, upper bound: 0.0124195
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036715, 0.0036505
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206795, 0.0205609
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144264, 0.0144954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0105217, upper bound: 0.0104430
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0105217, upper bound: 0.0104430
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036583, 0.0036352
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206434, 0.0204942
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0143796, 0.0144579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126574, upper bound: 0.0123776
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125404, upper bound: 0.0124599
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036591, 0.0036303
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206446, 0.0204845
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0143653, 0.0144595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129512, upper bound: 0.0124474
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129512, upper bound: 0.0124474
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037067, 0.0036870
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206858, 0.0205631
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145416, 0.0146070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125291, upper bound: 0.0122985
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125291, upper bound: 0.0122985
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037069, 0.0036869
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206772, 0.0205754
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145418, 0.0146071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128633, upper bound: 0.0126490
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129507, upper bound: 0.0124597
time: 1.12 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124544, upper bound: 0.0127880
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0123813, upper bound: 0.0128905
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124442, upper bound: 0.0127880
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0123779, upper bound: 0.0128917
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126490, upper bound: 0.0128633
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126344, upper bound: 0.0128631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126542, upper bound: 0.0128275
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126408, upper bound: 0.0128276
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0122954, upper bound: 0.0125337
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0122954, upper bound: 0.0125337
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125249, upper bound: 0.0127182
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124522, upper bound: 0.0128189
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124752, upper bound: 0.0129345
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124634, upper bound: 0.0129365
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124194, upper bound: 0.0124388
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0123409, upper bound: 0.0125592
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0119718, upper bound: 0.0123961
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0121700, upper bound: 0.0122877
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0119718, upper bound: 0.0123961
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0121700, upper bound: 0.0122877
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0119720, upper bound: 0.0124007
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0121672, upper bound: 0.0122881
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0119720, upper bound: 0.0124007
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0121672, upper bound: 0.0122881
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0123063, upper bound: 0.0125068
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0122968, upper bound: 0.0125069
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0119929, upper bound: 0.0123787
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0121759, upper bound: 0.0122560
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124746, upper bound: 0.0129345
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126537, upper bound: 0.0128281
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0122953, upper bound: 0.0125132
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0122953, upper bound: 0.0125132
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124731, upper bound: 0.0123709
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125952, upper bound: 0.0122124
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127237, upper bound: 0.0124989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127183, upper bound: 0.0125021
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127364, upper bound: 0.0124808
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127334, upper bound: 0.0124871
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125025, upper bound: 0.0123543
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126068, upper bound: 0.0121855
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125087, upper bound: 0.0123560
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125071, upper bound: 0.0123623
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126138, upper bound: 0.0121989
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126111, upper bound: 0.0122010
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126039, upper bound: 0.0126681
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127277, upper bound: 0.0124942
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126201, upper bound: 0.0125891
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126484, upper bound: 0.0125790
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126084, upper bound: 0.0126685
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126082, upper bound: 0.0126795
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127276, upper bound: 0.0124916
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127269, upper bound: 0.0124994
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0123680, upper bound: 0.0124620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124019, upper bound: 0.0124530
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126084, upper bound: 0.0126685
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126082, upper bound: 0.0126795
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127277, upper bound: 0.0124942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127270, upper bound: 0.0125012
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124895, upper bound: 0.0123010
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125166, upper bound: 0.0122801
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124994, upper bound: 0.0127269
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126795, upper bound: 0.0126082
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125012, upper bound: 0.0127270
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126795, upper bound: 0.0126038
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124916, upper bound: 0.0127276
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126685, upper bound: 0.0126084
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124942, upper bound: 0.0127277
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126681, upper bound: 0.0126039
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124865, upper bound: 0.0127323
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124874, upper bound: 0.0127334
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125021, upper bound: 0.0127183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125035, upper bound: 0.0127225
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0121989, upper bound: 0.0126138
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0122267, upper bound: 0.0126050
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125797, upper bound: 0.0127038
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125806, upper bound: 0.0127036
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0120188, upper bound: 0.0123318
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0120452, upper bound: 0.0123181
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0120113, upper bound: 0.0123321
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0121929, upper bound: 0.0122064
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0123306, upper bound: 0.0124623
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0123265, upper bound: 0.0124644
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0120113, upper bound: 0.0123321
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0121929, upper bound: 0.0122064
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125132, upper bound: 0.0122953
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125132, upper bound: 0.0122953
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125129, upper bound: 0.0123049
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125129, upper bound: 0.0123049
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0128189, upper bound: 0.0124522
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127182, upper bound: 0.0125250
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0122954
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0122954
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0122881, upper bound: 0.0121672
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0122881, upper bound: 0.0121672
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0129552, upper bound: 0.0124474
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0129549, upper bound: 0.0124598
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126250, upper bound: 0.0123940
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127317, upper bound: 0.0122266
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126245, upper bound: 0.0124006
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0127291, upper bound: 0.0122302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124423, upper bound: 0.0124195
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0124423, upper bound: 0.0124195
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0105217, upper bound: 0.0104430
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0105217, upper bound: 0.0104430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0126574, upper bound: 0.0123776
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125404, upper bound: 0.0124599
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0129512, upper bound: 0.0124474
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0129512, upper bound: 0.0124474
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125291, upper bound: 0.0122985
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0125291, upper bound: 0.0122985
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0128633, upper bound: 0.0126490
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0129507, upper bound: 0.0124597

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037235, 0.0037368
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0209212, 0.0210054
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147087, 0.0146644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122238, upper bound: 0.0125788
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122420, upper bound: 0.0125459
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037261, 0.0037394
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0209211, 0.0210054
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147160, 0.0146717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121440, upper bound: 0.0126812
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121682, upper bound: 0.0126561
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037180, 0.0037409
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208910, 0.0210264
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147229, 0.0146465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122125, upper bound: 0.0125788
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122341, upper bound: 0.0125465
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037206, 0.0037437
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0208910, 0.0210265
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0147310, 0.0146538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121399, upper bound: 0.0126815
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121648, upper bound: 0.0126588
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036362, 0.0036519
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204581, 0.0205514
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144438, 0.0143918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124082, upper bound: 0.0124781
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123237, upper bound: 0.0125840
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036313, 0.0036547
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204294, 0.0205686
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144534, 0.0143748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123945, upper bound: 0.0124781
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123154, upper bound: 0.0125839
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036376, 0.0036445
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204531, 0.0205311
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144213, 0.0143950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121663, upper bound: 0.0122424
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121663, upper bound: 0.0122424
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036338, 0.0036488
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204301, 0.0205516
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144352, 0.0143827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124040, upper bound: 0.0124365
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123306, upper bound: 0.0125525
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036848, 0.0037056
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205498, 0.0206481
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146008, 0.0145329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119622, upper bound: 0.0123891
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121567, upper bound: 0.0122765
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036848, 0.0037073
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205403, 0.0206794
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146082, 0.0145332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119622, upper bound: 0.0123891
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121567, upper bound: 0.0122765
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036784, 0.0037051
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205419, 0.0206964
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146032, 0.0145150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122125, upper bound: 0.0125788
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123929, upper bound: 0.0124715
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036810, 0.0037076
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205419, 0.0206964
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146106, 0.0145223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121432, upper bound: 0.0126831
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123155, upper bound: 0.0125832
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036333, 0.0036484
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204448, 0.0205390
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144324, 0.0143825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119839, upper bound: 0.0123710
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119839, upper bound: 0.0123710
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036292, 0.0036526
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204213, 0.0205585
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144470, 0.0143691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123716
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123716
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036531, 0.0036624
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205015, 0.0206055
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144754, 0.0144402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123366, upper bound: 0.0125521
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123308, upper bound: 0.0125525
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036871, 0.0036975
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205417, 0.0206338
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145768, 0.0145411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119836, upper bound: 0.0123645
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121663, upper bound: 0.0122424
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036832, 0.0037018
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205198, 0.0206543
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145909, 0.0145276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123649
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121564, upper bound: 0.0122416
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036333, 0.0036484
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204338, 0.0205482
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144323, 0.0143822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0122422, upper bound: 0.0125461
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121745, upper bound: 0.0126596
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036374, 0.0036449
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0204401, 0.0205437
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144230, 0.0143946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124150, upper bound: 0.0124331
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123364, upper bound: 0.0125521
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036829, 0.0037024
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205168, 0.0206591
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145928, 0.0145265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123716
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121540, upper bound: 0.0122460
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036826, 0.0037041
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205044, 0.0206904
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0146003, 0.0145250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123716
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121540, upper bound: 0.0122460
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036609, 0.0036489
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206080, 0.0204930
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144279, 0.0144714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124670, upper bound: 0.0123622
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124637, upper bound: 0.0123652
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036645, 0.0036450
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206113, 0.0204867
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144165, 0.0144817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125892, upper bound: 0.0122043
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125841, upper bound: 0.0122060
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037005, 0.0036806
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206748, 0.0205514
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145216, 0.0145892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124670, upper bound: 0.0123624
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125892, upper bound: 0.0122060
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036961, 0.0036844
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206541, 0.0205737
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145344, 0.0145754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124637, upper bound: 0.0123660
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125841, upper bound: 0.0122084
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037062, 0.0036766
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0207026, 0.0205318
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145093, 0.0146077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124956, upper bound: 0.0123440
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126016, upper bound: 0.0121863
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037035, 0.0036820
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206848, 0.0205600
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0145270, 0.0145983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124943, upper bound: 0.0123500
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125989, upper bound: 0.0121880
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036664, 0.0036474
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206247, 0.0204972
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144240, 0.0144886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124963, upper bound: 0.0123433
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124947, upper bound: 0.0123487
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036676, 0.0036430
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0206269, 0.0204880
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0144106, 0.0144912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126008, upper bound: 0.0121773
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125977, upper bound: 0.0121797
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036509, 0.0036264
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205693, 0.0204315
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0143613, 0.0144441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124956, upper bound: 0.0123440
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124963, upper bound: 0.0123433
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036482, 0.0036313
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205513, 0.0204595
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0143779, 0.0144347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124943, upper bound: 0.0123504
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124947, upper bound: 0.0123492
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036521, 0.0036214
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205713, 0.0204210
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0143471, 0.0144463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126016, upper bound: 0.0121873
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126008, upper bound: 0.0121785
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0036490, 0.0036269
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0205531, 0.0204504
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0143647, 0.0144358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125989, upper bound: 0.0121893
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125977, upper bound: 0.0121812
time: 1.20 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0122238, upper bound: 0.0125788
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0122420, upper bound: 0.0125459
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121440, upper bound: 0.0126812
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121682, upper bound: 0.0126561
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0122125, upper bound: 0.0125788
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0122341, upper bound: 0.0125465
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121399, upper bound: 0.0126815
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121648, upper bound: 0.0126588
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124082, upper bound: 0.0124781
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123237, upper bound: 0.0125840
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123945, upper bound: 0.0124781
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123154, upper bound: 0.0125839
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121663, upper bound: 0.0122424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121663, upper bound: 0.0122424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124040, upper bound: 0.0124365
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123306, upper bound: 0.0125525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119622, upper bound: 0.0123891
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121567, upper bound: 0.0122765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119622, upper bound: 0.0123891
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121567, upper bound: 0.0122765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0122125, upper bound: 0.0125788
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123929, upper bound: 0.0124715
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121432, upper bound: 0.0126831
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123155, upper bound: 0.0125832
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119839, upper bound: 0.0123710
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119839, upper bound: 0.0123710
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123716
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123716
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123366, upper bound: 0.0125521
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123308, upper bound: 0.0125525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119836, upper bound: 0.0123645
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121663, upper bound: 0.0122424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123649
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121564, upper bound: 0.0122416
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0122422, upper bound: 0.0125461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121745, upper bound: 0.0126596
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124150, upper bound: 0.0124331
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0123364, upper bound: 0.0125521
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121540, upper bound: 0.0122460
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0119767, upper bound: 0.0123716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0121540, upper bound: 0.0122460
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124670, upper bound: 0.0123622
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124637, upper bound: 0.0123652
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0125892, upper bound: 0.0122043
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0125841, upper bound: 0.0122060
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124670, upper bound: 0.0123624
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0125892, upper bound: 0.0122060
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124637, upper bound: 0.0123660
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0125841, upper bound: 0.0122084
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124956, upper bound: 0.0123440
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0126016, upper bound: 0.0121863
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124943, upper bound: 0.0123500
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0125989, upper bound: 0.0121880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124963, upper bound: 0.0123433
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124947, upper bound: 0.0123487
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0126008, upper bound: 0.0121773
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0125977, upper bound: 0.0121797
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124956, upper bound: 0.0123440
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124963, upper bound: 0.0123433
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124943, upper bound: 0.0123504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0124947, upper bound: 0.0123492
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0126016, upper bound: 0.0121873
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0126008, upper bound: 0.0121785
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0125989, upper bound: 0.0121893
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.04
Output dim: 8, lower bound: -0.0125977, upper bound: 0.0121812
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126039, upper bound: 0.0126681
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127277, upper bound: 0.0124942
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126201, upper bound: 0.0125891
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126484, upper bound: 0.0125790
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126084, upper bound: 0.0126685
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126082, upper bound: 0.0126795
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127276, upper bound: 0.0124916
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127269, upper bound: 0.0124994
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126084, upper bound: 0.0126685
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126082, upper bound: 0.0126795
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127277, upper bound: 0.0124942
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127270, upper bound: 0.0125012
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124895, upper bound: 0.0123010
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125166, upper bound: 0.0122801
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124994, upper bound: 0.0127269
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126795, upper bound: 0.0126082
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125012, upper bound: 0.0127270
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126795, upper bound: 0.0126038
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124916, upper bound: 0.0127276
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126685, upper bound: 0.0126084
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124942, upper bound: 0.0127277
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126681, upper bound: 0.0126039
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124865, upper bound: 0.0127323
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0124874, upper bound: 0.0127334
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125021, upper bound: 0.0127183
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125035, upper bound: 0.0127225
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0121989, upper bound: 0.0126138
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0122267, upper bound: 0.0126050
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125797, upper bound: 0.0127038
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125806, upper bound: 0.0127036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125132, upper bound: 0.0122953
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125132, upper bound: 0.0122953
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125129, upper bound: 0.0123049
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125129, upper bound: 0.0123049
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128189, upper bound: 0.0124522
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127182, upper bound: 0.0125250
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0122954
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125337, upper bound: 0.0122954
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129552, upper bound: 0.0124474
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129549, upper bound: 0.0124598
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126250, upper bound: 0.0123940
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127317, upper bound: 0.0122266
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126245, upper bound: 0.0124006
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0127291, upper bound: 0.0122302
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0126574, upper bound: 0.0123776
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125404, upper bound: 0.0124599
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129512, upper bound: 0.0124474
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129512, upper bound: 0.0124474
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125291, upper bound: 0.0122985
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0125291, upper bound: 0.0122985
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0128633, upper bound: 0.0126490
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 8, lower bound: -0.0129507, upper bound: 0.0124597

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.57 + 597.23 = 600.80 seconds
