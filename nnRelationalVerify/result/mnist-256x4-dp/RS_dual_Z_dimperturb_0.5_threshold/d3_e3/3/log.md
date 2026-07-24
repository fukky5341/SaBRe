## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00071487


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028950, 0.0028950)
1: (-0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0008162, 0.0008162)
2: (-0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0060221, 0.0060221)
3: (0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007969, 0.0007969)
4: (0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0045006, 0.0045006)
5: (0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012504, 0.0012504)
6: (0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0011350, 0.0011350)
7: (-0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0042355, 0.0042355)
8: (-0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0032965, 0.0032965)
9: (-0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002844, 0.0002844)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.68 = 3.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0007840, upper bound: 0.0007840

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007777, upper bound: 0.0007648
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007649, upper bound: 0.0007777
time: 0.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 5, lower bound: -0.0007777, upper bound: 0.0007648
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 5, lower bound: -0.0007649, upper bound: 0.0007777

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028806, 0.0028834
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0008122, 0.0008129
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0059922, 0.0059981
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007930, 0.0007937
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0044826, 0.0044782
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012454, 0.0012442
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0011304, 0.0011293
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0042186, 0.0042145
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0032802, 0.0032834
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002833, 0.0002830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007686, upper bound: 0.0007444
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007587, upper bound: 0.0007560
time: 0.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028834, 0.0028806
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0008129, 0.0008122
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0059981, 0.0059922
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007937, 0.0007930
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0044782, 0.0044826
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012442, 0.0012454
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0011293, 0.0011304
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0042145, 0.0042186
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0032834, 0.0032802
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002830, 0.0002833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007558, upper bound: 0.0007586
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007443, upper bound: 0.0007686
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 5, lower bound: -0.0007686, upper bound: 0.0007444
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 5, lower bound: -0.0007587, upper bound: 0.0007560
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 5, lower bound: -0.0007558, upper bound: 0.0007586
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 5, lower bound: -0.0007443, upper bound: 0.0007686

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027985, 0.0028198
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007890, 0.0007950
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058215, 0.0058657
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007704, 0.0007762
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043836, 0.0043506
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012179, 0.0012087
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0011055, 0.0010972
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0041255, 0.0040944
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031867, 0.0032109
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002770, 0.0002749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007568, upper bound: 0.0007208
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007464, upper bound: 0.0007324
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028172, 0.0028013
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007943, 0.0007898
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058604, 0.0058273
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007755, 0.0007712
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043550, 0.0043797
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012099, 0.0012168
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010983, 0.0011045
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0040985, 0.0041218
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0032080, 0.0031899
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002752, 0.0002768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007466, upper bound: 0.0007311
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007357, upper bound: 0.0007445
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028013, 0.0028172
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007898, 0.0007943
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058273, 0.0058604
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007712, 0.0007755
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043797, 0.0043550
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012168, 0.0012099
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0011045, 0.0010983
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0041218, 0.0040985
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031899, 0.0032080
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002768, 0.0002752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007445, upper bound: 0.0007357
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007312, upper bound: 0.0007466
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028198, 0.0027985
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007950, 0.0007890
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058657, 0.0058215
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007762, 0.0007704
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043506, 0.0043836
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012087, 0.0012179
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010972, 0.0011055
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0040944, 0.0041255
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0032109, 0.0031867
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002749, 0.0002770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007324, upper bound: 0.0007463
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007209, upper bound: 0.0007568
time: 0.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 5, lower bound: -0.0007568, upper bound: 0.0007208
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 5, lower bound: -0.0007464, upper bound: 0.0007324
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 5, lower bound: -0.0007466, upper bound: 0.0007311
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 5, lower bound: -0.0007357, upper bound: 0.0007445
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 5, lower bound: -0.0007445, upper bound: 0.0007357
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 5, lower bound: -0.0007312, upper bound: 0.0007466
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 5, lower bound: -0.0007324, upper bound: 0.0007463
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 5, lower bound: -0.0007209, upper bound: 0.0007568

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027790, 0.0028185
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007835, 0.0007946
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0057809, 0.0058631
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007650, 0.0007759
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043817, 0.0043203
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012174, 0.0012003
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0011050, 0.0010895
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0041237, 0.0040659
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031645, 0.0032095
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002769, 0.0002730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007446, upper bound: 0.0006955
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007266, upper bound: 0.0007067
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027985, 0.0028003
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007890, 0.0007895
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058215, 0.0058251
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007704, 0.0007709
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043533, 0.0043506
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012095, 0.0012087
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010978, 0.0010972
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0040970, 0.0040944
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031867, 0.0031887
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002751, 0.0002749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007350, upper bound: 0.0007083
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007111, upper bound: 0.0007177
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027977, 0.0027982
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007888, 0.0007889
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058198, 0.0058209
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007702, 0.0007703
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043501, 0.0043494
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012086, 0.0012084
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010970, 0.0010968
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0040940, 0.0040932
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031858, 0.0031863
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002749, 0.0002749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007343, upper bound: 0.0007041
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007152, upper bound: 0.0007175
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028172, 0.0027818
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007943, 0.0007843
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058604, 0.0057868
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007755, 0.0007658
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043247, 0.0043797
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012015, 0.0012168
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010906, 0.0011045
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0040700, 0.0041218
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0032080, 0.0031677
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002733, 0.0002768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007245, upper bound: 0.0007184
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007024, upper bound: 0.0007302
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027818, 0.0028158
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007843, 0.0007939
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0057868, 0.0058575
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007658, 0.0007751
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043775, 0.0043247
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012162, 0.0012015
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0011039, 0.0010906
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0041197, 0.0040700
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031677, 0.0032064
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002766, 0.0002733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007301, upper bound: 0.0007024
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007184, upper bound: 0.0007245
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028013, 0.0027977
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007898, 0.0007888
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058273, 0.0058198
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007712, 0.0007702
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043494, 0.0043550
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012084, 0.0012099
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010968, 0.0010983
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0040932, 0.0040985
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031899, 0.0031858
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002749, 0.0002752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007176, upper bound: 0.0007152
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007041, upper bound: 0.0007343
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028003, 0.0027956
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007895, 0.0007882
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058251, 0.0058155
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007709, 0.0007696
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043461, 0.0043533
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012075, 0.0012095
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010960, 0.0010978
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0040902, 0.0040970
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031887, 0.0031834
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002746, 0.0002751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007177, upper bound: 0.0007111
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007083, upper bound: 0.0007350
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0028198, 0.0027790
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007950, 0.0007835
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0058657, 0.0057809
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007762, 0.0007650
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0043203, 0.0043836
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0012003, 0.0012179
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010895, 0.0011055
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0040659, 0.0041255
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0032109, 0.0031645
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002730, 0.0002770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007067, upper bound: 0.0007267
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006955, upper bound: 0.0007445
time: 0.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007446, upper bound: 0.0006955
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007266, upper bound: 0.0007067
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007350, upper bound: 0.0007083
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007111, upper bound: 0.0007177
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007343, upper bound: 0.0007041
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007152, upper bound: 0.0007175
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007245, upper bound: 0.0007184
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007024, upper bound: 0.0007302
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007301, upper bound: 0.0007024
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007184, upper bound: 0.0007245
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007176, upper bound: 0.0007152
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007041, upper bound: 0.0007343
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007177, upper bound: 0.0007111
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007083, upper bound: 0.0007350
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0007067, upper bound: 0.0007267
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.74
Output dim: 5, lower bound: -0.0006955, upper bound: 0.0007445

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026623, 0.0027195
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007506, 0.0007667
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055381, 0.0056572
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007329, 0.0007486
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0042278, 0.0041389
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011746, 0.0011499
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010662, 0.0010438
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039789, 0.0038951
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030316, 0.0030967
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002672, 0.0002616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007351, upper bound: 0.0006784
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007198, upper bound: 0.0006842
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026803, 0.0027018
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007557, 0.0007617
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055756, 0.0056203
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007378, 0.0007438
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0042003, 0.0041668
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011670, 0.0011577
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010592, 0.0010508
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039529, 0.0039215
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030521, 0.0030766
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002654, 0.0002633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007162, upper bound: 0.0006847
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007085, upper bound: 0.0006961
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026831, 0.0027016
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007565, 0.0007617
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055815, 0.0056199
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007386, 0.0007437
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0042000, 0.0041713
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011669, 0.0011589
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010592, 0.0010519
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039526, 0.0039256
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030553, 0.0030763
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002654, 0.0002636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007253, upper bound: 0.0006879
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007102, upper bound: 0.0006977
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027011, 0.0026835
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007616, 0.0007566
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056189, 0.0055823
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007436, 0.0007387
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041719, 0.0041992
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011591, 0.0011667
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010521, 0.0010590
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039262, 0.0039519
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030758, 0.0030558
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002636, 0.0002654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007006, upper bound: 0.0006945
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006930, upper bound: 0.0007075
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026810, 0.0026987
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007559, 0.0007609
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055770, 0.0056138
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007380, 0.0007429
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041954, 0.0041679
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011656, 0.0011580
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010580, 0.0010511
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039483, 0.0039225
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030529, 0.0030730
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002651, 0.0002634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007253, upper bound: 0.0006867
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007074, upper bound: 0.0006931
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026987, 0.0026815
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007609, 0.0007560
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056138, 0.0055781
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007429, 0.0007382
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041687, 0.0041954
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011582, 0.0011656
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010513, 0.0010580
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039232, 0.0039483
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030730, 0.0030534
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002634, 0.0002651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007047, upper bound: 0.0006981
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006961, upper bound: 0.0007072
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027018, 0.0026837
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007617, 0.0007566
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056204, 0.0055826
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007438, 0.0007388
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041721, 0.0042003
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011591, 0.0011670
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010521, 0.0010593
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039264, 0.0039530
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030766, 0.0030559
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002636, 0.0002654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007154, upper bound: 0.0006977
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006968, upper bound: 0.0007080
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027195, 0.0026651
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007667, 0.0007514
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056571, 0.0055440
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007486, 0.0007337
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041432, 0.0042278
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011511, 0.0011746
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010449, 0.0010662
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038992, 0.0039788
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030967, 0.0030348
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002618, 0.0002672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006919, upper bound: 0.0007074
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006840, upper bound: 0.0007203
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026651, 0.0027165
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007514, 0.0007659
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055440, 0.0056510
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007337, 0.0007478
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0042232, 0.0041432
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011733, 0.0011511
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010650, 0.0010449
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039745, 0.0038992
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030348, 0.0030933
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002669, 0.0002618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007203, upper bound: 0.0006841
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007074, upper bound: 0.0006919
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026837, 0.0026991
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007566, 0.0007610
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055826, 0.0056147
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007388, 0.0007430
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041960, 0.0041721
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011658, 0.0011591
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010582, 0.0010521
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039489, 0.0039264
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030559, 0.0030735
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002652, 0.0002636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007080, upper bound: 0.0006969
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006977, upper bound: 0.0007155
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026859, 0.0026987
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007573, 0.0007609
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055873, 0.0056138
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007394, 0.0007429
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041954, 0.0041756
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011656, 0.0011601
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010580, 0.0010530
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039483, 0.0039297
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030585, 0.0030730
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002651, 0.0002639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007072, upper bound: 0.0006961
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006981, upper bound: 0.0007048
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027045, 0.0026810
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007625, 0.0007559
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056259, 0.0055770
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007445, 0.0007380
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041679, 0.0042045
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011580, 0.0011681
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010511, 0.0010603
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039225, 0.0039569
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030796, 0.0030529
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002634, 0.0002657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006932, upper bound: 0.0007074
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006866, upper bound: 0.0007253
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026835, 0.0026954
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007566, 0.0007599
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055823, 0.0056070
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007387, 0.0007420
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041903, 0.0041719
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011642, 0.0011591
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010567, 0.0010521
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039436, 0.0039262
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030558, 0.0030693
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002648, 0.0002636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007075, upper bound: 0.0006929
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006945, upper bound: 0.0007005
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027016, 0.0026789
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007617, 0.0007553
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056199, 0.0055727
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007437, 0.0007375
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041647, 0.0042000
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011571, 0.0011669
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010503, 0.0010592
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039194, 0.0039526
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030763, 0.0030505
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002632, 0.0002654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006978, upper bound: 0.0007102
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006879, upper bound: 0.0007253
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027044, 0.0026803
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007625, 0.0007557
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056257, 0.0055756
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007445, 0.0007378
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041668, 0.0042043
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011577, 0.0011681
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010508, 0.0010603
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039215, 0.0039567
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030795, 0.0030521
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002633, 0.0002657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006961, upper bound: 0.0007085
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006847, upper bound: 0.0007162
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027225, 0.0026623
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007676, 0.0007506
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056632, 0.0055381
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007494, 0.0007329
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041389, 0.0042324
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011499, 0.0011759
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010438, 0.0010673
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038951, 0.0039831
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0031001, 0.0030316
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002616, 0.0002675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006842, upper bound: 0.0007197
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006783, upper bound: 0.0007350
time: 0.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007351, upper bound: 0.0006784
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007198, upper bound: 0.0006842
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007162, upper bound: 0.0006847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007085, upper bound: 0.0006961
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007253, upper bound: 0.0006879
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007102, upper bound: 0.0006977
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007006, upper bound: 0.0006945
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006930, upper bound: 0.0007075
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007253, upper bound: 0.0006867
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007074, upper bound: 0.0006931
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007047, upper bound: 0.0006981
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006961, upper bound: 0.0007072
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007154, upper bound: 0.0006977
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006968, upper bound: 0.0007080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006919, upper bound: 0.0007074
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006840, upper bound: 0.0007203
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007203, upper bound: 0.0006841
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007074, upper bound: 0.0006919
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007080, upper bound: 0.0006969
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006977, upper bound: 0.0007155
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007072, upper bound: 0.0006961
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006981, upper bound: 0.0007048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006932, upper bound: 0.0007074
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006866, upper bound: 0.0007253
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0007075, upper bound: 0.0006929
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006945, upper bound: 0.0007005
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006978, upper bound: 0.0007102
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006879, upper bound: 0.0007253
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006961, upper bound: 0.0007085
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006847, upper bound: 0.0007162
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006842, upper bound: 0.0007197
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 5, lower bound: -0.0006783, upper bound: 0.0007350

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026314, 0.0027086
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007419, 0.0007637
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054738, 0.0056345
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007244, 0.0007456
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0042109, 0.0040907
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011699, 0.0011365
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010619, 0.0010316
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039629, 0.0038498
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029963, 0.0030843
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002661, 0.0002585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007308, upper bound: 0.0006723
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007282, upper bound: 0.0006740
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026504, 0.0026886
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007473, 0.0007580
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055135, 0.0055928
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007296, 0.0007401
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041797, 0.0041204
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011612, 0.0011448
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010541, 0.0010391
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039336, 0.0038778
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030181, 0.0030615
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002641, 0.0002604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007153, upper bound: 0.0006785
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007142, upper bound: 0.0006797
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026494, 0.0026889
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007470, 0.0007581
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055112, 0.0055935
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007293, 0.0007402
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041803, 0.0041187
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011614, 0.0011443
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010542, 0.0010387
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039341, 0.0038762
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030168, 0.0030619
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002642, 0.0002603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007117, upper bound: 0.0006788
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007087, upper bound: 0.0006804
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026514, 0.0026903
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007475, 0.0007585
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055154, 0.0055963
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007299, 0.0007406
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041823, 0.0041218
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011620, 0.0011452
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010547, 0.0010395
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039360, 0.0038791
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030191, 0.0030634
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002643, 0.0002605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007211, upper bound: 0.0006807
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007187, upper bound: 0.0006833
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026500, 0.0026889
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007471, 0.0007581
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055126, 0.0055934
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007295, 0.0007402
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041802, 0.0041198
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011614, 0.0011446
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010542, 0.0010390
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039340, 0.0038772
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030176, 0.0030619
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002642, 0.0002603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0006799
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007202, upper bound: 0.0006822
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026701, 0.0026735
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007528, 0.0007538
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055543, 0.0055614
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007350, 0.0007360
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041563, 0.0041509
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011547, 0.0011532
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010481, 0.0010468
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039115, 0.0039065
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030404, 0.0030443
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002627, 0.0002623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007113, upper bound: 0.0006900
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0006931
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027071, 0.0026342
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007632, 0.0007427
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056314, 0.0054796
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007452, 0.0007251
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040951, 0.0042086
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011377, 0.0011693
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010327, 0.0010613
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038539, 0.0039607
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030826, 0.0029995
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002588, 0.0002660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006796, upper bound: 0.0007139
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006777, upper bound: 0.0007160
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026342, 0.0027052
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007427, 0.0007627
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054796, 0.0056273
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007251, 0.0007447
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0042055, 0.0040951
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011684, 0.0011377
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010606, 0.0010327
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039579, 0.0038539
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029995, 0.0030804
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002658, 0.0002588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007160, upper bound: 0.0006777
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007139, upper bound: 0.0006796
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026735, 0.0026681
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007538, 0.0007522
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055614, 0.0055503
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007360, 0.0007345
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041479, 0.0041563
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011524, 0.0011547
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010460, 0.0010481
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039037, 0.0039115
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030443, 0.0030382
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002621, 0.0002627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006931, upper bound: 0.0007098
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006900, upper bound: 0.0007113
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026935, 0.0026500
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007594, 0.0007471
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056030, 0.0055126
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007415, 0.0007295
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041198, 0.0041874
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011446, 0.0011634
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010390, 0.0010560
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038772, 0.0039408
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030671, 0.0030176
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002603, 0.0002646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006822, upper bound: 0.0007202
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006799, upper bound: 0.0007211
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026903, 0.0026480
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007585, 0.0007466
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055963, 0.0055083
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007406, 0.0007289
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041166, 0.0041823
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011437, 0.0011620
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010381, 0.0010547
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038742, 0.0039360
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030634, 0.0030153
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002601, 0.0002643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006834, upper bound: 0.0007187
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006808, upper bound: 0.0007211
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026903, 0.0026494
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007585, 0.0007470
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055964, 0.0055112
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007406, 0.0007293
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041187, 0.0041824
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011443, 0.0011620
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010387, 0.0010547
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038762, 0.0039361
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030635, 0.0030168
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002603, 0.0002643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006804, upper bound: 0.0007087
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006789, upper bound: 0.0007117
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026907, 0.0026505
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007586, 0.0007473
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055971, 0.0055135
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007407, 0.0007296
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041204, 0.0041830
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011448, 0.0011621
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010391, 0.0010549
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038778, 0.0039366
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030639, 0.0030181
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002604, 0.0002643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006797, upper bound: 0.0007143
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006785, upper bound: 0.0007153
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027103, 0.0026314
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007641, 0.0007419
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056379, 0.0054738
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007461, 0.0007244
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040907, 0.0042134
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011365, 0.0011706
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010316, 0.0010626
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038498, 0.0039653
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030862, 0.0029963
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002585, 0.0002663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006740, upper bound: 0.0007283
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006723, upper bound: 0.0007307
time: 0.95 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007308, upper bound: 0.0006723
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007282, upper bound: 0.0006740
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007153, upper bound: 0.0006785
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007142, upper bound: 0.0006797
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007117, upper bound: 0.0006788
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007087, upper bound: 0.0006804
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007211, upper bound: 0.0006807
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007187, upper bound: 0.0006833
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0006799
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007202, upper bound: 0.0006822
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007113, upper bound: 0.0006900
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0006931
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006796, upper bound: 0.0007139
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006777, upper bound: 0.0007160
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007160, upper bound: 0.0006777
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0007139, upper bound: 0.0006796
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006931, upper bound: 0.0007098
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006900, upper bound: 0.0007113
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006822, upper bound: 0.0007202
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006799, upper bound: 0.0007211
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006834, upper bound: 0.0007187
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006808, upper bound: 0.0007211
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006804, upper bound: 0.0007087
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006789, upper bound: 0.0007117
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006797, upper bound: 0.0007143
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006785, upper bound: 0.0007153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006740, upper bound: 0.0007283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 5, lower bound: -0.0006723, upper bound: 0.0007307

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0025862, 0.0026725
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007291, 0.0007535
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0053798, 0.0055594
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007119, 0.0007357
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041547, 0.0040206
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011543, 0.0011170
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010478, 0.0010139
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039101, 0.0037838
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029449, 0.0030432
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002626, 0.0002541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007303, upper bound: 0.0006704
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007285, upper bound: 0.0006719
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0025952, 0.0026615
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007317, 0.0007504
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0053986, 0.0055364
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007144, 0.0007326
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041375, 0.0040346
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011495, 0.0011209
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010434, 0.0010175
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038939, 0.0037970
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029552, 0.0030306
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002615, 0.0002550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007279, upper bound: 0.0006722
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007259, upper bound: 0.0006735
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026038, 0.0026525
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007341, 0.0007478
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054164, 0.0055176
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007168, 0.0007302
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041235, 0.0040479
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011456, 0.0011246
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010399, 0.0010208
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038807, 0.0038095
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029649, 0.0030204
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002606, 0.0002558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007149, upper bound: 0.0006766
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0006781
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026060, 0.0026541
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007347, 0.0007483
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054210, 0.0055212
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007174, 0.0007306
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041262, 0.0040513
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011464, 0.0011256
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010406, 0.0010217
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038832, 0.0038127
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029675, 0.0030223
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002607, 0.0002560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007206, upper bound: 0.0006783
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007186, upper bound: 0.0006804
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026150, 0.0026434
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007373, 0.0007453
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054397, 0.0054989
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007199, 0.0007277
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041095, 0.0040653
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011418, 0.0011295
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010364, 0.0010252
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038675, 0.0038259
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029777, 0.0030101
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002597, 0.0002569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007182, upper bound: 0.0006809
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007166, upper bound: 0.0006829
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026049, 0.0026528
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007344, 0.0007479
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054187, 0.0055183
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007171, 0.0007303
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041240, 0.0040496
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011458, 0.0011251
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010400, 0.0010213
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038812, 0.0038111
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029662, 0.0030207
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002606, 0.0002559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007207, upper bound: 0.0006787
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007197, upper bound: 0.0006795
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026139, 0.0026430
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007370, 0.0007452
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054375, 0.0054980
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007196, 0.0007276
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041089, 0.0040636
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011416, 0.0011290
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010362, 0.0010248
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038669, 0.0038243
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029765, 0.0030096
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002597, 0.0002568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007198, upper bound: 0.0006803
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007184, upper bound: 0.0006817
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026708, 0.0025891
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007530, 0.0007300
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055558, 0.0053859
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007352, 0.0007127
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040251, 0.0041520
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011183, 0.0011536
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010151, 0.0010471
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0037881, 0.0039075
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030412, 0.0029483
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002544, 0.0002624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006772, upper bound: 0.0007096
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006762, upper bound: 0.0007155
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0025891, 0.0026691
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007300, 0.0007525
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0053859, 0.0055522
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007127, 0.0007347
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041494, 0.0040251
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011528, 0.0011183
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010464, 0.0010151
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039050, 0.0037881
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029483, 0.0030393
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002622, 0.0002544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007155, upper bound: 0.0006762
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007096, upper bound: 0.0006772
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026469, 0.0026139
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007463, 0.0007370
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055062, 0.0054375
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007287, 0.0007196
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040636, 0.0041150
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011290, 0.0011433
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010248, 0.0010377
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038243, 0.0038726
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030141, 0.0029765
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002568, 0.0002600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006818, upper bound: 0.0007184
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006803, upper bound: 0.0007197
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026572, 0.0026049
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007492, 0.0007344
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055274, 0.0054187
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007315, 0.0007171
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040496, 0.0041308
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011251, 0.0011477
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010213, 0.0010417
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038111, 0.0038876
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030257, 0.0029662
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002559, 0.0002610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006795, upper bound: 0.0007197
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006787, upper bound: 0.0007207
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026434, 0.0026118
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007453, 0.0007364
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054989, 0.0054332
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007277, 0.0007190
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040604, 0.0041095
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011281, 0.0011418
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010240, 0.0010364
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038213, 0.0038675
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030101, 0.0029741
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002566, 0.0002597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006829, upper bound: 0.0007166
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006809, upper bound: 0.0007182
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026541, 0.0026015
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007483, 0.0007335
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055212, 0.0054116
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007306, 0.0007161
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040443, 0.0041262
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011236, 0.0011464
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010199, 0.0010406
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038062, 0.0038832
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030223, 0.0029623
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002556, 0.0002607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006804, upper bound: 0.0007186
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006784, upper bound: 0.0007206
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026543, 0.0026038
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007484, 0.0007341
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055215, 0.0054164
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007307, 0.0007168
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040479, 0.0041264
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011246, 0.0011464
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010208, 0.0010406
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038095, 0.0038834
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030225, 0.0029649
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002558, 0.0002608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006781, upper bound: 0.0007134
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006766, upper bound: 0.0007149
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026632, 0.0025952
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007509, 0.0007317
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055401, 0.0053986
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007331, 0.0007144
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040346, 0.0041403
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011209, 0.0011503
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010175, 0.0010441
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0037970, 0.0038965
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030326, 0.0029552
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002550, 0.0002616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006735, upper bound: 0.0007259
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006723, upper bound: 0.0007279
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026739, 0.0025862
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007539, 0.0007291
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055623, 0.0053798
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007361, 0.0007119
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040206, 0.0041569
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011170, 0.0011549
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010139, 0.0010483
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0037838, 0.0039121
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030448, 0.0029449
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002541, 0.0002627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006719, upper bound: 0.0007285
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006705, upper bound: 0.0007303
time: 0.99 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007303, upper bound: 0.0006704
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007285, upper bound: 0.0006719
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007279, upper bound: 0.0006722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007259, upper bound: 0.0006735
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007149, upper bound: 0.0006766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0006781
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007206, upper bound: 0.0006783
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007186, upper bound: 0.0006804
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007182, upper bound: 0.0006809
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007166, upper bound: 0.0006829
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007207, upper bound: 0.0006787
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007197, upper bound: 0.0006795
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007198, upper bound: 0.0006803
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007184, upper bound: 0.0006817
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006772, upper bound: 0.0007096
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006762, upper bound: 0.0007155
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007155, upper bound: 0.0006762
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0007096, upper bound: 0.0006772
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006818, upper bound: 0.0007184
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006803, upper bound: 0.0007197
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006795, upper bound: 0.0007197
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006787, upper bound: 0.0007207
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006829, upper bound: 0.0007166
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006809, upper bound: 0.0007182
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006804, upper bound: 0.0007186
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006784, upper bound: 0.0007206
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006781, upper bound: 0.0007134
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006766, upper bound: 0.0007149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006735, upper bound: 0.0007259
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006723, upper bound: 0.0007279
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006719, upper bound: 0.0007285
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.83
Output dim: 5, lower bound: -0.0006705, upper bound: 0.0007303

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026120, 0.0027008
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007364, 0.0007614
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054334, 0.0056181
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007190, 0.0007435
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041986, 0.0040606
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011665, 0.0011281
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010588, 0.0010240
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039514, 0.0038215
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029742, 0.0030754
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002653, 0.0002566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006936, upper bound: 0.0006377
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006936, upper bound: 0.0006377
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026145, 0.0026975
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007371, 0.0007605
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054386, 0.0056114
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007197, 0.0007426
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041936, 0.0040645
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011651, 0.0011292
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010576, 0.0010250
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039467, 0.0038251
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029771, 0.0030717
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002650, 0.0002568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006919, upper bound: 0.0006395
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006919, upper bound: 0.0006395
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026203, 0.0026897
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007388, 0.0007583
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054508, 0.0055951
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007213, 0.0007404
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041814, 0.0040736
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011617, 0.0011318
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010545, 0.0010273
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039352, 0.0038337
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029838, 0.0030628
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002642, 0.0002574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006905, upper bound: 0.0006396
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006905, upper bound: 0.0006396
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026235, 0.0026869
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007397, 0.0007575
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054574, 0.0055893
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007222, 0.0007397
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041771, 0.0040785
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011605, 0.0011331
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010534, 0.0010285
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039311, 0.0038383
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029874, 0.0030596
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002640, 0.0002577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006415
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006414
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026328, 0.0026824
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007423, 0.0007563
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054768, 0.0055799
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007248, 0.0007384
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041701, 0.0040930
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011586, 0.0011372
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010516, 0.0010322
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039245, 0.0038520
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029980, 0.0030545
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002635, 0.0002587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006889, upper bound: 0.0006396
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006889, upper bound: 0.0006396
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026353, 0.0026784
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007430, 0.0007551
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054820, 0.0055716
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007255, 0.0007373
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041639, 0.0040969
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011569, 0.0011382
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010501, 0.0010332
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039187, 0.0038557
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030009, 0.0030499
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002631, 0.0002589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006872, upper bound: 0.0006422
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006872, upper bound: 0.0006422
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026412, 0.0026717
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007447, 0.0007533
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054942, 0.0055577
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007271, 0.0007355
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041535, 0.0041060
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011540, 0.0011408
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010474, 0.0010355
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039089, 0.0038642
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030075, 0.0030423
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002625, 0.0002595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006866, upper bound: 0.0006421
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006866, upper bound: 0.0006421
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026444, 0.0026689
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007455, 0.0007525
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055008, 0.0055518
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007279, 0.0007347
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041491, 0.0041109
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011527, 0.0011421
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010463, 0.0010367
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039047, 0.0038689
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030111, 0.0030391
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002622, 0.0002598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006850, upper bound: 0.0006443
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006850, upper bound: 0.0006442
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026300, 0.0026810
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007415, 0.0007559
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054708, 0.0055771
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007240, 0.0007380
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041679, 0.0040886
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011580, 0.0011359
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010511, 0.0010311
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039225, 0.0038478
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029947, 0.0030529
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002634, 0.0002584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006821, upper bound: 0.0006445
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006821, upper bound: 0.0006445
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026331, 0.0026781
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007424, 0.0007551
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054775, 0.0055711
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007249, 0.0007372
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041635, 0.0040935
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011567, 0.0011373
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010500, 0.0010323
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039183, 0.0038525
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029984, 0.0030496
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002631, 0.0002587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006815, upper bound: 0.0006463
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006815, upper bound: 0.0006463
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026383, 0.0026713
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007438, 0.0007531
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054882, 0.0055568
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007263, 0.0007354
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041528, 0.0041016
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011538, 0.0011395
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010473, 0.0010344
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039083, 0.0038600
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030043, 0.0030418
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002624, 0.0002592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006800, upper bound: 0.0006468
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006800, upper bound: 0.0006468
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026422, 0.0026695
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007449, 0.0007526
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054962, 0.0055531
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007273, 0.0007349
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041501, 0.0041076
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011530, 0.0011412
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010466, 0.0010359
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039057, 0.0038657
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030087, 0.0030398
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002623, 0.0002596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006796, upper bound: 0.0006487
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006796, upper bound: 0.0006487
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027001, 0.0026156
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007613, 0.0007374
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056168, 0.0054410
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007433, 0.0007200
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040663, 0.0041977
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011297, 0.0011662
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010255, 0.0010586
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038268, 0.0039505
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030747, 0.0029784
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002570, 0.0002653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006443, upper bound: 0.0006768
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006443, upper bound: 0.0006768
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026156, 0.0026973
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007374, 0.0007605
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0054410, 0.0056110
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007200, 0.0007425
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041933, 0.0040663
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011650, 0.0011297
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010575, 0.0010255
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0039463, 0.0038268
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0029784, 0.0030714
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002650, 0.0002570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006768, upper bound: 0.0006443
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006768, upper bound: 0.0006443
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026745, 0.0026422
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007540, 0.0007449
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055635, 0.0054962
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007362, 0.0007273
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041076, 0.0041578
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011412, 0.0011552
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010359, 0.0010485
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038657, 0.0039129
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030454, 0.0030087
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002596, 0.0002627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006487, upper bound: 0.0006796
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006487, upper bound: 0.0006796
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026763, 0.0026383
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007545, 0.0007438
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055672, 0.0054882
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007367, 0.0007263
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041016, 0.0041606
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011395, 0.0011559
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010344, 0.0010492
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038600, 0.0039156
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030475, 0.0030043
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002592, 0.0002629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006468, upper bound: 0.0006800
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006468, upper bound: 0.0006799
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026830, 0.0026331
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007564, 0.0007424
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055811, 0.0054775
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007386, 0.0007249
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040935, 0.0041710
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011373, 0.0011588
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010323, 0.0010519
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038525, 0.0039253
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030551, 0.0029984
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002587, 0.0002636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006463, upper bound: 0.0006815
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006463, upper bound: 0.0006815
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026865, 0.0026300
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007574, 0.0007415
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055885, 0.0054708
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007395, 0.0007240
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040886, 0.0041765
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011359, 0.0011603
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010311, 0.0010532
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038478, 0.0039305
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030591, 0.0029947
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002584, 0.0002639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006445, upper bound: 0.0006821
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006445, upper bound: 0.0006820
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026689, 0.0026401
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007525, 0.0007443
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055518, 0.0054919
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007347, 0.0007268
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041043, 0.0041491
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011403, 0.0011527
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010351, 0.0010463
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038626, 0.0039047
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030391, 0.0030063
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002594, 0.0002622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006443, upper bound: 0.0006849
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006443, upper bound: 0.0006849
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026717, 0.0026375
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007533, 0.0007436
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055577, 0.0054865
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007355, 0.0007261
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0041003, 0.0041535
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011392, 0.0011540
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010340, 0.0010474
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038588, 0.0039089
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030423, 0.0030033
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002591, 0.0002625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006421, upper bound: 0.0006866
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006421, upper bound: 0.0006866
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026784, 0.0026297
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007551, 0.0007414
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055716, 0.0054704
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007373, 0.0007239
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040882, 0.0041639
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011358, 0.0011569
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010310, 0.0010501
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038475, 0.0039187
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030499, 0.0029945
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002584, 0.0002631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006422, upper bound: 0.0006872
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006422, upper bound: 0.0006872
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026824, 0.0026274
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007563, 0.0007408
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055799, 0.0054655
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007384, 0.0007233
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040846, 0.0041701
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011348, 0.0011586
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010301, 0.0010516
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038440, 0.0039245
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030545, 0.0029918
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002581, 0.0002635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006396, upper bound: 0.0006889
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006396, upper bound: 0.0006889
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026837, 0.0026293
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007566, 0.0007413
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055826, 0.0054694
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007388, 0.0007238
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040875, 0.0041721
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011356, 0.0011591
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010308, 0.0010521
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038468, 0.0039264
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030559, 0.0029940
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002583, 0.0002636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006447, upper bound: 0.0006775
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006447, upper bound: 0.0006775
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026897, 0.0026235
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007583, 0.0007397
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0055952, 0.0054574
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007404, 0.0007222
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040785, 0.0041815
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011331, 0.0011617
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010285, 0.0010545
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038383, 0.0039353
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030628, 0.0029874
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002577, 0.0002642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006414, upper bound: 0.0006888
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006414, upper bound: 0.0006888
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026926, 0.0026203
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007591, 0.0007388
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056011, 0.0054508
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007412, 0.0007213
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040736, 0.0041859
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011318, 0.0011630
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010273, 0.0010556
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038337, 0.0039394
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030660, 0.0029838
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002574, 0.0002645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006396, upper bound: 0.0006905
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006396, upper bound: 0.0006905
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0026993, 0.0026145
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007610, 0.0007371
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056151, 0.0054386
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007431, 0.0007197
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040645, 0.0041964
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011292, 0.0011659
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010250, 0.0010583
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038251, 0.0039492
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030737, 0.0029771
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002568, 0.0002652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006395, upper bound: 0.0006919
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006395, upper bound: 0.0006919
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0137610, -0.0096707, -0.0137610, -0.0096707, -0.0027033, 0.0026120
1: -0.0068184, -0.0056652, -0.0068184, -0.0056652, -0.0007622, 0.0007364
2: -0.0117478, -0.0032392, -0.0117478, -0.0032392, -0.0056233, 0.0054334
3: 0.0000727, 0.0011986, 0.0000727, 0.0011986, -0.0007442, 0.0007190
4: 0.0085126, 0.0148714, 0.0085126, 0.0148714, -0.0040606, 0.0042025
5: 0.9978713, 0.9996380, 0.9978713, 0.9996380, -0.0011281, 0.0011676
6: 0.0059514, 0.0075550, 0.0059514, 0.0075550, -0.0010240, 0.0010598
7: -0.0011718, 0.0048126, -0.0011718, 0.0048126, -0.0038215, 0.0039551
8: -0.0129385, -0.0082808, -0.0129385, -0.0082808, -0.0030782, 0.0029742
9: -0.0032953, -0.0028935, -0.0032953, -0.0028935, -0.0002566, 0.0002656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006377, upper bound: 0.0006936
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006377, upper bound: 0.0006936
time: 1.09 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006936, upper bound: 0.0006377
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006936, upper bound: 0.0006377
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006919, upper bound: 0.0006395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006919, upper bound: 0.0006395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006905, upper bound: 0.0006396
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006905, upper bound: 0.0006396
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006415
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006889, upper bound: 0.0006396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006889, upper bound: 0.0006396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006872, upper bound: 0.0006422
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006872, upper bound: 0.0006422
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006866, upper bound: 0.0006421
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006866, upper bound: 0.0006421
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006850, upper bound: 0.0006443
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006850, upper bound: 0.0006442
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006821, upper bound: 0.0006445
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006821, upper bound: 0.0006445
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006815, upper bound: 0.0006463
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006815, upper bound: 0.0006463
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006800, upper bound: 0.0006468
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006800, upper bound: 0.0006468
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006796, upper bound: 0.0006487
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006796, upper bound: 0.0006487
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006443, upper bound: 0.0006768
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006443, upper bound: 0.0006768
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006768, upper bound: 0.0006443
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006768, upper bound: 0.0006443
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006487, upper bound: 0.0006796
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006487, upper bound: 0.0006796
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006468, upper bound: 0.0006800
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006468, upper bound: 0.0006799
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006463, upper bound: 0.0006815
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006463, upper bound: 0.0006815
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006445, upper bound: 0.0006821
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006445, upper bound: 0.0006820
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006443, upper bound: 0.0006849
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006443, upper bound: 0.0006849
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006421, upper bound: 0.0006866
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006421, upper bound: 0.0006866
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006422, upper bound: 0.0006872
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006422, upper bound: 0.0006872
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006396, upper bound: 0.0006889
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006396, upper bound: 0.0006889
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006447, upper bound: 0.0006775
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006447, upper bound: 0.0006775
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006414, upper bound: 0.0006888
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006414, upper bound: 0.0006888
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006396, upper bound: 0.0006905
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006396, upper bound: 0.0006905
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006395, upper bound: 0.0006919
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006395, upper bound: 0.0006919
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006377, upper bound: 0.0006936
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.00
Output dim: 5, lower bound: -0.0006377, upper bound: 0.0006936

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.27 + 327.84 = 331.11 seconds
