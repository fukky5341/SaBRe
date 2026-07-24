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
execution time: IAR + RelationalAnalysis = 1.17 + 2.42 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0138512, upper bound: 0.0138512

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132910, upper bound: 0.0132910
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132910, upper bound: 0.0132910
time: 1.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 8, lower bound: -0.0132910, upper bound: 0.0132910
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 8, lower bound: -0.0132910, upper bound: 0.0132910

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0039510, 0.0039510
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0221178, 0.0221078
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0153879, 0.0153878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130505, upper bound: 0.0130661
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130661, upper bound: 0.0130505
time: 1.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0039510, 0.0039527
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0221078, 0.0221395
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0153951, 0.0153879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130505, upper bound: 0.0130661
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130661, upper bound: 0.0130505
time: 1.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 8, lower bound: -0.0130505, upper bound: 0.0130661
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 8, lower bound: -0.0130661, upper bound: 0.0130505
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 8, lower bound: -0.0130505, upper bound: 0.0130661
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.76
Output dim: 8, lower bound: -0.0130661, upper bound: 0.0130505

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038550, 0.0038566
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0216186, 0.0216073
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0150830, 0.0150781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127678, upper bound: 0.0129272
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129134, upper bound: 0.0127914
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038570, 0.0038549
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0216168, 0.0216087
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0150781, 0.0150851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127914, upper bound: 0.0129134
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129272, upper bound: 0.0127678
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038549, 0.0038583
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0216087, 0.0216391
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0150903, 0.0150781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127678, upper bound: 0.0129272
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129134, upper bound: 0.0127914
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038566, 0.0038567
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0216073, 0.0216404
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0150854, 0.0150830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127914, upper bound: 0.0129134
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129272, upper bound: 0.0127678
time: 1.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0127678, upper bound: 0.0129272
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0129134, upper bound: 0.0127914
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0127914, upper bound: 0.0129134
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0129272, upper bound: 0.0127678
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0127678, upper bound: 0.0129272
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0129134, upper bound: 0.0127914
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0127914, upper bound: 0.0129134
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0129272, upper bound: 0.0127678

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038100, 0.0038156
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0215048, 0.0214998
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149571, 0.0149405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126976, upper bound: 0.0128548
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126976, upper bound: 0.0128548
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038144, 0.0038116
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0215142, 0.0214934
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149455, 0.0149544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128385, upper bound: 0.0127223
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128385, upper bound: 0.0127223
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038120, 0.0038143
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0215029, 0.0215039
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149539, 0.0149476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127223, upper bound: 0.0128385
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127223, upper bound: 0.0128385
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038160, 0.0038099
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0215085, 0.0214948
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149406, 0.0149589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128548, upper bound: 0.0126976
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128548, upper bound: 0.0126976
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038099, 0.0038173
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214948, 0.0215315
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149644, 0.0149406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126976, upper bound: 0.0128548
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126976, upper bound: 0.0128548
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038143, 0.0038133
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0215039, 0.0215252
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149528, 0.0149539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128385, upper bound: 0.0127223
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128385, upper bound: 0.0127223
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038116, 0.0038160
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214934, 0.0215356
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149611, 0.0149455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127223, upper bound: 0.0128385
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127223, upper bound: 0.0128385
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038156, 0.0038117
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214998, 0.0215265
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149479, 0.0149571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128548, upper bound: 0.0126976
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128548, upper bound: 0.0126976
time: 1.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0126976, upper bound: 0.0128548
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0126976, upper bound: 0.0128548
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0128385, upper bound: 0.0127223
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0128385, upper bound: 0.0127223
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0127223, upper bound: 0.0128385
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0127223, upper bound: 0.0128385
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0128548, upper bound: 0.0126976
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0128548, upper bound: 0.0126976
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0126976, upper bound: 0.0128548
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0126976, upper bound: 0.0128548
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0128385, upper bound: 0.0127223
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0128385, upper bound: 0.0127223
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0127223, upper bound: 0.0128385
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0127223, upper bound: 0.0128385
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0128548, upper bound: 0.0126976
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 8, lower bound: -0.0128548, upper bound: 0.0126976

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038060, 0.0038115
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214462, 0.0214297
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149520, 0.0149357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124407, upper bound: 0.0126587
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125007, upper bound: 0.0126116
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038061, 0.0038116
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214347, 0.0214406
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149523, 0.0149357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124407, upper bound: 0.0126587
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125007, upper bound: 0.0126116
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038104, 0.0038075
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214552, 0.0214234
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149403, 0.0149495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126003, upper bound: 0.0125309
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126451, upper bound: 0.0124619
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038104, 0.0038076
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214441, 0.0214345
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149406, 0.0149494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126003, upper bound: 0.0125309
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126451, upper bound: 0.0124619
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038080, 0.0038104
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214466, 0.0214338
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149490, 0.0149427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124619, upper bound: 0.0126451
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125309, upper bound: 0.0126003
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038079, 0.0038103
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214328, 0.0214452
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149490, 0.0149425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124619, upper bound: 0.0126451
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125309, upper bound: 0.0126003
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038120, 0.0038060
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214524, 0.0214247
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149356, 0.0149540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126116, upper bound: 0.0125007
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126587, upper bound: 0.0124407
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038119, 0.0038060
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214384, 0.0214362
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149358, 0.0149539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126116, upper bound: 0.0125007
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126587, upper bound: 0.0124407
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038060, 0.0038132
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214362, 0.0214614
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149593, 0.0149358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124407, upper bound: 0.0126587
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125007, upper bound: 0.0126116
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038060, 0.0038133
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214247, 0.0214724
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149596, 0.0149356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124407, upper bound: 0.0126587
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125007, upper bound: 0.0126116
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038103, 0.0038092
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214452, 0.0214551
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149476, 0.0149490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126003, upper bound: 0.0125309
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126451, upper bound: 0.0124619
time: 1.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038104, 0.0038093
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214338, 0.0214662
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149479, 0.0149490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126003, upper bound: 0.0125309
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126451, upper bound: 0.0124620
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038076, 0.0038121
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214345, 0.0214655
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149563, 0.0149406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124619, upper bound: 0.0126451
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125309, upper bound: 0.0126003
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038075, 0.0038120
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214234, 0.0214769
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149563, 0.0149403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124619, upper bound: 0.0126451
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125309, upper bound: 0.0126003
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038116, 0.0038077
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214407, 0.0214564
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149429, 0.0149523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126116, upper bound: 0.0125007
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126587, upper bound: 0.0124407
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038115, 0.0038077
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214297, 0.0214679
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149430, 0.0149520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126116, upper bound: 0.0125007
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126587, upper bound: 0.0124407
time: 1.17 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0124407, upper bound: 0.0126587
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0125007, upper bound: 0.0126116
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0124407, upper bound: 0.0126587
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0125007, upper bound: 0.0126116
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126003, upper bound: 0.0125309
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126451, upper bound: 0.0124619
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126003, upper bound: 0.0125309
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126451, upper bound: 0.0124619
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0124619, upper bound: 0.0126451
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0125309, upper bound: 0.0126003
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0124619, upper bound: 0.0126451
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0125309, upper bound: 0.0126003
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126116, upper bound: 0.0125007
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126587, upper bound: 0.0124407
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126116, upper bound: 0.0125007
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126587, upper bound: 0.0124407
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0124407, upper bound: 0.0126587
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0125007, upper bound: 0.0126116
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0124407, upper bound: 0.0126587
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0125007, upper bound: 0.0126116
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126003, upper bound: 0.0125309
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126451, upper bound: 0.0124619
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126003, upper bound: 0.0125309
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126451, upper bound: 0.0124620
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0124619, upper bound: 0.0126451
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0125309, upper bound: 0.0126003
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0124619, upper bound: 0.0126451
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0125309, upper bound: 0.0126003
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126116, upper bound: 0.0125007
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126587, upper bound: 0.0124407
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126116, upper bound: 0.0125007
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 8, lower bound: -0.0126587, upper bound: 0.0124407

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037955, 0.0038015
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214164, 0.0213998
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149262, 0.0149086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119825, upper bound: 0.0124130
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121930, upper bound: 0.0122228
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037960, 0.0038010
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214163, 0.0213990
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149250, 0.0149099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0123436
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122701, upper bound: 0.0121861
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037956, 0.0038016
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214046, 0.0214108
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149264, 0.0149086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119825, upper bound: 0.0124130
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121930, upper bound: 0.0122228
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037961, 0.0038011
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214048, 0.0214098
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149252, 0.0149099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0123436
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122701, upper bound: 0.0121861
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037999, 0.0037977
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214251, 0.0213935
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149148, 0.0149225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121802, upper bound: 0.0123003
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123317, upper bound: 0.0120557
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038002, 0.0037970
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214254, 0.0213930
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149133, 0.0149233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122135, upper bound: 0.0122176
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123952, upper bound: 0.0120041
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037999, 0.0037977
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214138, 0.0214046
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149151, 0.0149223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121802, upper bound: 0.0123003
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123317, upper bound: 0.0120557
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038003, 0.0037971
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214143, 0.0214041
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149136, 0.0149232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122135, upper bound: 0.0122176
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123952, upper bound: 0.0120041
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037975, 0.0038003
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214175, 0.0214040
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149231, 0.0149157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120041, upper bound: 0.0123952
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122176, upper bound: 0.0122135
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037981, 0.0037999
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214168, 0.0214030
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149219, 0.0149170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120557, upper bound: 0.0123317
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123003, upper bound: 0.0121802
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037974, 0.0038002
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214039, 0.0214154
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149230, 0.0149155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120041, upper bound: 0.0123952
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122176, upper bound: 0.0122135
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037981, 0.0037998
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214030, 0.0214148
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149220, 0.0149169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120557, upper bound: 0.0123317
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123003, upper bound: 0.0121802
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038015, 0.0037961
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214233, 0.0213948
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149100, 0.0149270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121861, upper bound: 0.0122701
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123436, upper bound: 0.0120302
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038020, 0.0037955
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214226, 0.0213939
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149086, 0.0149280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122228, upper bound: 0.0121930
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124130, upper bound: 0.0119825
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038014, 0.0037961
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214094, 0.0214063
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149102, 0.0149268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121861, upper bound: 0.0122701
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123436, upper bound: 0.0120302
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038019, 0.0037955
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214085, 0.0214063
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149087, 0.0149278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122228, upper bound: 0.0121930
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124130, upper bound: 0.0119825
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037955, 0.0038032
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214063, 0.0214314
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149335, 0.0149087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119825, upper bound: 0.0124130
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121930, upper bound: 0.0122228
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037961, 0.0038027
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214063, 0.0214306
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149323, 0.0149102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0123436
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122701, upper bound: 0.0121861
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037955, 0.0038033
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0213939, 0.0214423
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149337, 0.0149086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119825, upper bound: 0.0124130
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121930, upper bound: 0.0122228
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037961, 0.0038028
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0213948, 0.0214413
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149325, 0.0149101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0123436
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122701, upper bound: 0.0121861
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037998, 0.0037995
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214148, 0.0214250
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149221, 0.0149220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121802, upper bound: 0.0123003
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123317, upper bound: 0.0120557
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038002, 0.0037988
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214154, 0.0214245
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149206, 0.0149230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122135, upper bound: 0.0122176
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123952, upper bound: 0.0120041
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037999, 0.0037994
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214030, 0.0214361
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149223, 0.0149219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121802, upper bound: 0.0123003
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123317, upper bound: 0.0120557
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038003, 0.0037988
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214040, 0.0214356
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149209, 0.0149231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122135, upper bound: 0.0122176
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123952, upper bound: 0.0120041
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037971, 0.0038021
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214041, 0.0214355
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149303, 0.0149136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120041, upper bound: 0.0123952
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122176, upper bound: 0.0122135
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037977, 0.0038016
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214046, 0.0214345
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149292, 0.0149151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120557, upper bound: 0.0123317
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123003, upper bound: 0.0121802
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037970, 0.0038019
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0213930, 0.0214469
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149303, 0.0149133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120041, upper bound: 0.0123952
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122176, upper bound: 0.0122135
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0037977, 0.0038015
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0213935, 0.0214463
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149292, 0.0149148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120557, upper bound: 0.0123317
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123003, upper bound: 0.0121802
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038011, 0.0037979
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214098, 0.0214264
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149173, 0.0149252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121861, upper bound: 0.0122701
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123436, upper bound: 0.0120302
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038016, 0.0037972
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0214108, 0.0214255
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149158, 0.0149264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122228, upper bound: 0.0121930
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124130, upper bound: 0.0119825
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038010, 0.0037978
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0213990, 0.0214378
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149175, 0.0149250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121861, upper bound: 0.0122701
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0123436, upper bound: 0.0120302
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986
1: -0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912
2: 0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394
3: -0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711
4: -0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0038015, 0.0037972
5: -0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973
6: -0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037
7: -0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0213998, 0.0214378
8: 0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723
9: -0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0149160, 0.0149262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0122228, upper bound: 0.0121930
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0124130, upper bound: 0.0119825
time: 1.09 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0119825, upper bound: 0.0124130
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121930, upper bound: 0.0122228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0123436
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122701, upper bound: 0.0121861
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0119825, upper bound: 0.0124130
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121930, upper bound: 0.0122228
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0123436
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122701, upper bound: 0.0121861
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121802, upper bound: 0.0123003
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123317, upper bound: 0.0120557
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122135, upper bound: 0.0122176
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123952, upper bound: 0.0120041
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121802, upper bound: 0.0123003
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123317, upper bound: 0.0120557
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122135, upper bound: 0.0122176
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123952, upper bound: 0.0120041
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120041, upper bound: 0.0123952
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122176, upper bound: 0.0122135
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120557, upper bound: 0.0123317
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123003, upper bound: 0.0121802
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120041, upper bound: 0.0123952
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122176, upper bound: 0.0122135
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120557, upper bound: 0.0123317
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123003, upper bound: 0.0121802
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121861, upper bound: 0.0122701
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123436, upper bound: 0.0120302
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122228, upper bound: 0.0121930
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0124130, upper bound: 0.0119825
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121861, upper bound: 0.0122701
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123436, upper bound: 0.0120302
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122228, upper bound: 0.0121930
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0124130, upper bound: 0.0119825
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0119825, upper bound: 0.0124130
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121930, upper bound: 0.0122228
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0123436
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122701, upper bound: 0.0121861
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0119825, upper bound: 0.0124130
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121930, upper bound: 0.0122228
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120302, upper bound: 0.0123436
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122701, upper bound: 0.0121861
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121802, upper bound: 0.0123003
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123317, upper bound: 0.0120557
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122135, upper bound: 0.0122176
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123952, upper bound: 0.0120041
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121802, upper bound: 0.0123003
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123317, upper bound: 0.0120557
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122135, upper bound: 0.0122176
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123952, upper bound: 0.0120041
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120041, upper bound: 0.0123952
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122176, upper bound: 0.0122135
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120557, upper bound: 0.0123317
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123003, upper bound: 0.0121802
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120041, upper bound: 0.0123952
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122176, upper bound: 0.0122135
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0120557, upper bound: 0.0123317
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123003, upper bound: 0.0121802
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121861, upper bound: 0.0122701
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123436, upper bound: 0.0120302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122228, upper bound: 0.0121930
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0124130, upper bound: 0.0119825
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0121861, upper bound: 0.0122701
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0123436, upper bound: 0.0120302
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0122228, upper bound: 0.0121930
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 8, lower bound: -0.0124130, upper bound: 0.0119825

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.59 + 226.85 = 230.44 seconds
