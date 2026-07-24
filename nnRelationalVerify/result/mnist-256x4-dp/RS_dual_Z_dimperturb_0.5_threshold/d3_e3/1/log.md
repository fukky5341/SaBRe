## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00504846


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0063162, 0.0063162)
1: (0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0009125, 0.0009125)
2: (0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0034921, 0.0034921)
3: (-0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0036117, 0.0036117)
4: (-0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0039099, 0.0039099)
5: (0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0037000, 0.0037000)
6: (-0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0146807, 0.0146807)
7: (-0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0199938, 0.0199938)
8: (0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0140841, 0.0140841)
9: (-0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0127846, 0.0127846)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.69 + 1.95 = 3.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0080077, upper bound: 0.0080077

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074859, upper bound: 0.0074859
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0074859, upper bound: 0.0074859
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 8, lower bound: -0.0074859, upper bound: 0.0074859
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 8, lower bound: -0.0074859, upper bound: 0.0074859

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0062799, 0.0063090
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0009073, 0.0009115
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0034881, 0.0034720
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0036075, 0.0035909
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0038874, 0.0039053
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0036958, 0.0036787
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0146638, 0.0145962
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0198787, 0.0199708
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0140030, 0.0140678
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0127698, 0.0127110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071212, upper bound: 0.0071584
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071584, upper bound: 0.0071212
time: 0.87 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0063162, 0.0062799
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0009125, 0.0009073
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0034720, 0.0034921
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0035909, 0.0036117
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0039099, 0.0038874
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0036787, 0.0037000
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0145962, 0.0146807
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0199938, 0.0198787
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0140841, 0.0140030
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0127110, 0.0127846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071212, upper bound: 0.0071584
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071584, upper bound: 0.0071212
time: 0.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 8, lower bound: -0.0071212, upper bound: 0.0071584
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 8, lower bound: -0.0071584, upper bound: 0.0071212
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 8, lower bound: -0.0071212, upper bound: 0.0071584
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 8, lower bound: -0.0071584, upper bound: 0.0071212

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0061495, 0.0062053
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008884, 0.0008965
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0034307, 0.0033999
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0035482, 0.0035163
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0038066, 0.0038412
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0036350, 0.0036023
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0144228, 0.0142931
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0194659, 0.0196426
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0137122, 0.0138367
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0125600, 0.0124470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065985, upper bound: 0.0066103
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065985, upper bound: 0.0066103
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0061707, 0.0061785
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008915, 0.0008926
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0034160, 0.0034116
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0035329, 0.0035285
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0038198, 0.0038246
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0036194, 0.0036148
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0143606, 0.0143424
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0195331, 0.0195579
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0137595, 0.0137770
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0125059, 0.0124900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066103, upper bound: 0.0065985
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066103, upper bound: 0.0065985
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0061853, 0.0061707
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008936, 0.0008915
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0034116, 0.0034197
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0035285, 0.0035368
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0038288, 0.0038198
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0036148, 0.0036233
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0143424, 0.0143762
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0195792, 0.0195331
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0137920, 0.0137595
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0124900, 0.0125195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065985, upper bound: 0.0066103
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065985, upper bound: 0.0066103
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0062065, 0.0061495
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008967, 0.0008884
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033999, 0.0034314
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0035163, 0.0035489
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0038419, 0.0038066
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0036023, 0.0036357
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0142931, 0.0144255
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0196463, 0.0194659
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0138393, 0.0137122
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0124470, 0.0125624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066103, upper bound: 0.0065985
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066103, upper bound: 0.0065985
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 8, lower bound: -0.0065985, upper bound: 0.0066103
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 8, lower bound: -0.0065985, upper bound: 0.0066103
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 8, lower bound: -0.0066103, upper bound: 0.0065985
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 8, lower bound: -0.0066103, upper bound: 0.0065985
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 8, lower bound: -0.0065985, upper bound: 0.0066103
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 8, lower bound: -0.0065985, upper bound: 0.0066103
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 8, lower bound: -0.0066103, upper bound: 0.0065985
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 8, lower bound: -0.0066103, upper bound: 0.0065985

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0060338, 0.0060891
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008717, 0.0008797
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033665, 0.0033359
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0034818, 0.0034502
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0037350, 0.0037693
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0035670, 0.0035346
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0141528, 0.0140242
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0190998, 0.0192749
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0134543, 0.0135777
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0123249, 0.0122129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0060333, 0.0061000
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008716, 0.0008813
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033725, 0.0033357
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0034880, 0.0034499
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0037347, 0.0037760
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0035734, 0.0035343
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0141780, 0.0140231
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0190982, 0.0193093
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0134532, 0.0136018
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0123469, 0.0122119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0060590, 0.0060624
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008753, 0.0008758
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033517, 0.0033499
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0034665, 0.0034646
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0037506, 0.0037527
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0035513, 0.0035494
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0140907, 0.0140828
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0191795, 0.0191902
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0135105, 0.0135180
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0122708, 0.0122639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0060545, 0.0060695
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008747, 0.0008769
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033556, 0.0033474
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0034706, 0.0034620
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0037479, 0.0037571
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0035555, 0.0035467
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0141071, 0.0140724
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0191654, 0.0192127
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0135005, 0.0135338
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0122851, 0.0122549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0060696, 0.0060545
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008769, 0.0008747
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033474, 0.0033557
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0034620, 0.0034706
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0037572, 0.0037479
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0035467, 0.0035556
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0140724, 0.0141074
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0192130, 0.0191654
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0135341, 0.0135005
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0122549, 0.0122853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0060691, 0.0060590
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008768, 0.0008754
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033499, 0.0033554
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0034646, 0.0034704
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0037569, 0.0037506
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0035494, 0.0035553
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0140828, 0.0141062
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0192115, 0.0191795
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0135330, 0.0135105
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0122639, 0.0122843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0060948, 0.0060333
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008805, 0.0008716
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033357, 0.0033696
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0034499, 0.0034850
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0037728, 0.0037347
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0035343, 0.0035703
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0140231, 0.0141659
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0192928, 0.0190982
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0135902, 0.0134532
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0122119, 0.0123363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0036021, 0.0100208, 0.0036021, 0.0100208, -0.0060903, 0.0060338
1: 0.0018427, 0.0027700, 0.0018427, 0.0027700, -0.0008799, 0.0008717
2: 0.0088196, 0.0123683, 0.0088196, 0.0123683, -0.0033359, 0.0033672
3: -0.0055588, -0.0018885, -0.0055588, -0.0018885, -0.0034502, 0.0034825
4: -0.0019925, 0.0019807, -0.0019925, 0.0019807, -0.0037700, 0.0037350
5: 0.0022389, 0.0059989, 0.0022389, 0.0059989, -0.0035346, 0.0035677
6: -0.0134170, 0.0015017, -0.0134170, 0.0015017, -0.0140242, 0.0141556
7: -0.0046018, 0.0157161, -0.0046018, 0.0157161, -0.0192786, 0.0190998
8: 0.9859722, 1.0002846, 0.9859722, 1.0002846, -0.0135803, 0.0134543
9: -0.0161457, -0.0031538, -0.0161457, -0.0031538, -0.0122129, 0.0123273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
time: 0.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0049388, upper bound: 0.0049388

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.63 + 45.16 = 48.79 seconds
