## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.03428451


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172260, 0.0172260)
1: (-0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110811, 0.0110811)
2: (0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0438499, 0.0438499)
3: (-0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0209962, 0.0209962)
4: (-0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393)
5: (0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821)
6: (-0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823)
7: (-0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0338952, 0.0338952)
8: (0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674)
9: (-0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.81 = 3.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0414153, upper bound: 0.0414153

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0389864, upper bound: 0.0389864
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0389864, upper bound: 0.0389864
time: 0.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.99
Output dim: 8, lower bound: -0.0389864, upper bound: 0.0389864
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.99
Output dim: 8, lower bound: -0.0389864, upper bound: 0.0389864

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172238, 0.0172127
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110811, 0.0110299
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0437567, 0.0438347
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0209261, 0.0209848
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0338704, 0.0337433
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0388733, upper bound: 0.0388719
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0388719, upper bound: 0.0388733
time: 0.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172127, 0.0172260
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110299, 0.0110811
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0438499, 0.0437567
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0209962, 0.0209261
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0337433, 0.0338952
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0388326, upper bound: 0.0388326
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0388326, upper bound: 0.0388326
time: 1.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.67 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 8, lower bound: -0.0388733, upper bound: 0.0388719
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 8, lower bound: -0.0388719, upper bound: 0.0388733
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 8, lower bound: -0.0388326, upper bound: 0.0388326
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 8, lower bound: -0.0388326, upper bound: 0.0388326

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171998, 0.0171866
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109655, 0.0109035
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435431, 0.0436360
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207669, 0.0208368
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0335440, 0.0333927
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0388522, upper bound: 0.0388424
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0388427, upper bound: 0.0388507
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171977, 0.0171885
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109555, 0.0109122
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435561, 0.0436211
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207767, 0.0208256
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0335198, 0.0334140
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0377798, upper bound: 0.0380097
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0380079, upper bound: 0.0377903
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172090, 0.0172216
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110120, 0.0110712
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0438185, 0.0437300
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0209727, 0.0209061
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0336999, 0.0338441
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370751, upper bound: 0.0370759
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370759, upper bound: 0.0370751
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172083, 0.0172223
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110087, 0.0110745
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0438236, 0.0437249
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0209764, 0.0209023
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0336916, 0.0338524
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0385415, upper bound: 0.0385378
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0385378, upper bound: 0.0385415
time: 0.92 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0388522, upper bound: 0.0388424
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0388427, upper bound: 0.0388507
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0377798, upper bound: 0.0380097
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0380079, upper bound: 0.0377903
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0370751, upper bound: 0.0370759
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0370759, upper bound: 0.0370751
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0385415, upper bound: 0.0385378
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 8, lower bound: -0.0385378, upper bound: 0.0385415

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171966, 0.0171818
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109441, 0.0108746
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435088, 0.0436129
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207362, 0.0208144
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334916, 0.0333221
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0385048, upper bound: 0.0384914
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0385048, upper bound: 0.0384914
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171950, 0.0171829
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109366, 0.0108796
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435163, 0.0436017
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207418, 0.0208060
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334734, 0.0333342
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0383663, upper bound: 0.0383991
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0383937, upper bound: 0.0383720
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171653, 0.0171684
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108383, 0.0108528
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434142, 0.0433925
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206733, 0.0206570
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331550, 0.0331904
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0377498, upper bound: 0.0379781
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0377438, upper bound: 0.0379871
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171977, 0.0171560
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109555, 0.0107950
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433276, 0.0436211
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206082, 0.0208256
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0335198, 0.0330492
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0374942, upper bound: 0.0373356
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0375039, upper bound: 0.0372997
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171683, 0.0171905
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108328, 0.0109372
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435957, 0.0434394
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0208021, 0.0206846
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332259, 0.0334806
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370327, upper bound: 0.0369911
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369885, upper bound: 0.0370345
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172090, 0.0171808
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110120, 0.0108920
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435280, 0.0437300
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207511, 0.0209061
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0336999, 0.0333702
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367383, upper bound: 0.0367383
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367383, upper bound: 0.0367383
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171751, 0.0171876
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108367, 0.0108964
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435568, 0.0434674
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207759, 0.0207086
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332720, 0.0334177
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0384911, upper bound: 0.0384342
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0384365, upper bound: 0.0384847
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171736, 0.0171889
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108301, 0.0109023
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435657, 0.0434575
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207825, 0.0207011
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332558, 0.0334321
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0380887, upper bound: 0.0381224
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0381204, upper bound: 0.0380940
time: 1.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0385048, upper bound: 0.0384914
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0385048, upper bound: 0.0384914
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0383663, upper bound: 0.0383991
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0383937, upper bound: 0.0383720
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0377498, upper bound: 0.0379781
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0377438, upper bound: 0.0379871
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0374942, upper bound: 0.0373356
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0375039, upper bound: 0.0372997
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0370327, upper bound: 0.0369911
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0369885, upper bound: 0.0370345
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0367383, upper bound: 0.0367383
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0367383, upper bound: 0.0367383
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0384911, upper bound: 0.0384342
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0384365, upper bound: 0.0384847
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0380887, upper bound: 0.0381224
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.46
Output dim: 8, lower bound: -0.0381204, upper bound: 0.0380940

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171951, 0.0171800
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109362, 0.0108654
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434960, 0.0436022
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207264, 0.0208062
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334729, 0.0333000
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0374646, upper bound: 0.0375969
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0376205, upper bound: 0.0374339
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171966, 0.0171803
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109441, 0.0108668
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434981, 0.0436129
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207280, 0.0208144
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334916, 0.0333033
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0383873, upper bound: 0.0383778
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0383867, upper bound: 0.0383778
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171766, 0.0171664
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108555, 0.0108076
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434007, 0.0434723
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206580, 0.0207119
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332680, 0.0331512
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0382241, upper bound: 0.0382435
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0382241, upper bound: 0.0382435
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171950, 0.0171644
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109366, 0.0107985
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433869, 0.0436017
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206477, 0.0208060
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334734, 0.0331289
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0380809, upper bound: 0.0380506
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0380809, upper bound: 0.0380506
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171621, 0.0171635
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108161, 0.0108230
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433795, 0.0433691
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206407, 0.0206330
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331033, 0.0331202
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0376783, upper bound: 0.0378650
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0376417, upper bound: 0.0379177
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171605, 0.0171648
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108085, 0.0108287
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433879, 0.0433578
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206471, 0.0206244
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330848, 0.0331340
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0373077, upper bound: 0.0375943
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0373272, upper bound: 0.0375664
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171792, 0.0171394
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108746, 0.0107237
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432116, 0.0434917
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205244, 0.0207317
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0333155, 0.0328712
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371514, upper bound: 0.0370591
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371514, upper bound: 0.0370591
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171977, 0.0171375
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109555, 0.0107151
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431987, 0.0436211
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205147, 0.0208256
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0335198, 0.0328502
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371669, upper bound: 0.0370279
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371669, upper bound: 0.0370279
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171342, 0.0171554
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105156, 0.0106169
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432442, 0.0430925
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205062, 0.0203921
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325989, 0.0328462
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0769741, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356740, upper bound: 0.0356224
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356740, upper bound: 0.0356224
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171332, 0.0171545
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105107, 0.0106128
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432381, 0.0430851
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205016, 0.0203866
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325870, 0.0328362
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0769398, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366404, upper bound: 0.0367567
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367131, upper bound: 0.0367116
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172075, 0.0171790
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110043, 0.0108827
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435153, 0.0437195
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207414, 0.0208978
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0336809, 0.0333482
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366957, upper bound: 0.0366673
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366673, upper bound: 0.0366957
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172090, 0.0171793
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110120, 0.0108840
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0435173, 0.0437300
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207430, 0.0209061
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0336999, 0.0333515
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362999, upper bound: 0.0363422
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363422, upper bound: 0.0362999
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171411, 0.0171526
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105335, 0.0105893
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432097, 0.0431261
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204902, 0.0204274
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326728, 0.0328090
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0772593, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0384669, upper bound: 0.0384072
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0384611, upper bound: 0.0384093
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171400, 0.0171537
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105283, 0.0105942
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432170, 0.0431184
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204957, 0.0204216
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326603, 0.0328209
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0772233, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0382851, upper bound: 0.0383266
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0382832, upper bound: 0.0383279
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171552, 0.0171730
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107490, 0.0108334
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434556, 0.0433293
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207033, 0.0206083
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330465, 0.0332524
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0376496, upper bound: 0.0377178
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0376869, upper bound: 0.0376940
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171736, 0.0171704
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108301, 0.0108213
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434375, 0.0434575
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206896, 0.0207011
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332558, 0.0332228
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0376858, upper bound: 0.0376918
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0377161, upper bound: 0.0376642
time: 0.98 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0374646, upper bound: 0.0375969
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0376205, upper bound: 0.0374339
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0383873, upper bound: 0.0383778
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0383867, upper bound: 0.0383778
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0382241, upper bound: 0.0382435
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0382241, upper bound: 0.0382435
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0380809, upper bound: 0.0380506
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0380809, upper bound: 0.0380506
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0376783, upper bound: 0.0378650
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0376417, upper bound: 0.0379177
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0373077, upper bound: 0.0375943
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0373272, upper bound: 0.0375664
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0371514, upper bound: 0.0370591
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0371514, upper bound: 0.0370591
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0371669, upper bound: 0.0370279
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0371669, upper bound: 0.0370279
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0356740, upper bound: 0.0356224
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0356740, upper bound: 0.0356224
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0366404, upper bound: 0.0367567
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0367131, upper bound: 0.0367116
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0366957, upper bound: 0.0366673
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0366673, upper bound: 0.0366957
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0362999, upper bound: 0.0363422
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0363422, upper bound: 0.0362999
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0384669, upper bound: 0.0384072
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0384611, upper bound: 0.0384093
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0382851, upper bound: 0.0383266
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0382832, upper bound: 0.0383279
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0376496, upper bound: 0.0377178
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0376869, upper bound: 0.0376940
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0376858, upper bound: 0.0376918
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 8, lower bound: -0.0377161, upper bound: 0.0376642

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171627, 0.0171610
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108182, 0.0108102
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433611, 0.0433731
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206267, 0.0206357
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331090, 0.0330894
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356275, upper bound: 0.0356984
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356394, upper bound: 0.0356984
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171951, 0.0171475
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109362, 0.0107473
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432670, 0.0436022
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205559, 0.0208062
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334729, 0.0329361
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357003, upper bound: 0.0356334
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357003, upper bound: 0.0356228
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171930, 0.0171759
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109270, 0.0108459
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434668, 0.0435874
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207044, 0.0207952
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334500, 0.0332523
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0383396, upper bound: 0.0382857
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0382949, upper bound: 0.0383270
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171922, 0.0171766
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109232, 0.0108492
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434718, 0.0435816
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0207082, 0.0207909
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334406, 0.0332605
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0379631, upper bound: 0.0379718
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0379784, upper bound: 0.0379541
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171730, 0.0171619
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108388, 0.0107867
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433694, 0.0434473
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206345, 0.0206931
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332273, 0.0331002
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371167, upper bound: 0.0373675
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0373511, upper bound: 0.0371613
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171721, 0.0171623
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108346, 0.0107887
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433723, 0.0434410
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206367, 0.0206884
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332170, 0.0331050
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0379304, upper bound: 0.0379631
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0379304, upper bound: 0.0379631
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171935, 0.0171627
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109288, 0.0107895
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433746, 0.0435910
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206383, 0.0207978
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334546, 0.0331077
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0320790, upper bound: 0.0320051
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0320790, upper bound: 0.0320051
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171950, 0.0171629
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109366, 0.0107906
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433762, 0.0436017
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206395, 0.0208060
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334734, 0.0331103
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370152, upper bound: 0.0371280
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371513, upper bound: 0.0369818
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171262, 0.0171273
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105107, 0.0105157
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430141, 0.0430068
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203397, 0.0203342
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324568, 0.0324688
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0767702, 0.0768045
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371864, upper bound: 0.0373761
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0372281, upper bound: 0.0373586
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171258, 0.0171302
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105088, 0.0105295
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430348, 0.0430038
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203553, 0.0203319
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324520, 0.0325025
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0767563, 0.0769012
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0316091, upper bound: 0.0316548
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0316091, upper bound: 0.0316548
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171239, 0.0171346
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106385, 0.0106886
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431243, 0.0430493
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204440, 0.0203875
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326016, 0.0327239
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0368713, upper bound: 0.0373011
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370201, upper bound: 0.0372188
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171320, 0.0171282
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106765, 0.0106586
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430795, 0.0431063
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204102, 0.0204304
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326945, 0.0326508
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361056, upper bound: 0.0363637
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361056, upper bound: 0.0363637
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171778, 0.0171376
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108666, 0.0107147
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431990, 0.0434810
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205144, 0.0207235
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332969, 0.0328495
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0310693, upper bound: 0.0311117
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0310693, upper bound: 0.0311117
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171792, 0.0171379
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108746, 0.0107160
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432009, 0.0434917
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205159, 0.0207317
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0333155, 0.0328526
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371280, upper bound: 0.0370152
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371225, upper bound: 0.0370293
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171962, 0.0171357
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109476, 0.0107057
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431856, 0.0436103
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205043, 0.0208174
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0335011, 0.0328276
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371445, upper bound: 0.0369822
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371380, upper bound: 0.0369992
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171977, 0.0171360
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109555, 0.0107073
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431880, 0.0436211
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205061, 0.0208256
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0335198, 0.0328315
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355305, upper bound: 0.0353483
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355305, upper bound: 0.0353483
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171275, 0.0171488
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104829, 0.0105848
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431981, 0.0430454
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204691, 0.0203544
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325179, 0.0327666
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0767135, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0345971, upper bound: 0.0347000
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347561, upper bound: 0.0345383
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171342, 0.0171487
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105156, 0.0105842
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431972, 0.0430925
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204685, 0.0203921
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325989, 0.0327652
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0769741, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0353558, upper bound: 0.0353645
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354149, upper bound: 0.0352934
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170937, 0.0171218
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102905, 0.0104245
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429174, 0.0427168
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202523, 0.0201014
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319673, 0.0322943
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0747424, 0.0756793
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352937, upper bound: 0.0354099
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352937, upper bound: 0.0354099
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170959, 0.0171156
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103008, 0.0103955
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428739, 0.0427321
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202196, 0.0201130
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319923, 0.0322234
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0748143, 0.0754763
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364014, upper bound: 0.0363735
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364014, upper bound: 0.0363735
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171736, 0.0171439
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106979, 0.0105627
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431639, 0.0433734
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204456, 0.0206129
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330747, 0.0327144
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0772979
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363537, upper bound: 0.0363256
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363499, upper bound: 0.0363269
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171725, 0.0171452
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106930, 0.0105686
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431729, 0.0433661
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204523, 0.0206074
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330627, 0.0327289
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773395
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366452, upper bound: 0.0366714
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366444, upper bound: 0.0366735
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171906, 0.0171631
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109310, 0.0108176
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0434065, 0.0436018
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206619, 0.0208132
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334906, 0.0331749
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0283211, upper bound: 0.0284098
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0283211, upper bound: 0.0284098
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0172090, 0.0171609
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0110120, 0.0108070
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433906, 0.0437300
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206500, 0.0209061
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0336999, 0.0331490
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360213, upper bound: 0.0360283
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360710, upper bound: 0.0359841
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171365, 0.0171476
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105037, 0.0105559
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431623, 0.0430841
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204510, 0.0203921
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325962, 0.0327237
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0770422, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367486, upper bound: 0.0366868
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367506, upper bound: 0.0366767
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171361, 0.0171486
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105020, 0.0105607
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431695, 0.0430816
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204564, 0.0203903
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325921, 0.0327354
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0770305, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0381104, upper bound: 0.0381217
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0381656, upper bound: 0.0380478
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171132, 0.0171265
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103982, 0.0104641
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430060, 0.0429073
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203313, 0.0202570
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0322927, 0.0324536
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0762598, 0.0767206
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0378432, upper bound: 0.0379299
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0378924, upper bound: 0.0378766
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171128, 0.0171275
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103965, 0.0104690
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430134, 0.0429048
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203368, 0.0202551
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0322886, 0.0324656
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0762480, 0.0767550
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0378292, upper bound: 0.0379210
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0378763, upper bound: 0.0378749
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171177, 0.0171396
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105683, 0.0106688
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431703, 0.0430197
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204781, 0.0203648
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325442, 0.0327896
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0771831, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363179, upper bound: 0.0363676
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363179, upper bound: 0.0363676
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171253, 0.0171357
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106040, 0.0106506
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431430, 0.0430733
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204575, 0.0204051
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326315, 0.0327451
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0376316, upper bound: 0.0375891
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0375796, upper bound: 0.0376392
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171361, 0.0171390
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106524, 0.0106663
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431665, 0.0431452
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204752, 0.0204591
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0327649, 0.0327834
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359367, upper bound: 0.0359979
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359373, upper bound: 0.0359967
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171438, 0.0171331
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106882, 0.0106385
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431249, 0.0431988
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204439, 0.0204994
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328522, 0.0327155
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0373801, upper bound: 0.0373657
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0374260, upper bound: 0.0373211
time: 1.11 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0356275, upper bound: 0.0356984
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0356394, upper bound: 0.0356984
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0357003, upper bound: 0.0356334
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0357003, upper bound: 0.0356228
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0383396, upper bound: 0.0382857
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0382949, upper bound: 0.0383270
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0379631, upper bound: 0.0379718
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0379784, upper bound: 0.0379541
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0371167, upper bound: 0.0373675
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0373511, upper bound: 0.0371613
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0379304, upper bound: 0.0379631
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0379304, upper bound: 0.0379631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0320790, upper bound: 0.0320051
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0320790, upper bound: 0.0320051
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0370152, upper bound: 0.0371280
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0371513, upper bound: 0.0369818
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0371864, upper bound: 0.0373761
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0372281, upper bound: 0.0373586
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0316091, upper bound: 0.0316548
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0316091, upper bound: 0.0316548
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0368713, upper bound: 0.0373011
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0370201, upper bound: 0.0372188
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0361056, upper bound: 0.0363637
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0361056, upper bound: 0.0363637
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0310693, upper bound: 0.0311117
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0310693, upper bound: 0.0311117
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0371280, upper bound: 0.0370152
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0371225, upper bound: 0.0370293
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0371445, upper bound: 0.0369822
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0371380, upper bound: 0.0369992
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0355305, upper bound: 0.0353483
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0355305, upper bound: 0.0353483
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0345971, upper bound: 0.0347000
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0347561, upper bound: 0.0345383
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0353558, upper bound: 0.0353645
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0354149, upper bound: 0.0352934
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0352937, upper bound: 0.0354099
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0352937, upper bound: 0.0354099
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0364014, upper bound: 0.0363735
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0364014, upper bound: 0.0363735
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0363537, upper bound: 0.0363256
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0363499, upper bound: 0.0363269
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0366452, upper bound: 0.0366714
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0366444, upper bound: 0.0366735
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0283211, upper bound: 0.0284098
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0283211, upper bound: 0.0284098
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0360213, upper bound: 0.0360283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0360710, upper bound: 0.0359841
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0367486, upper bound: 0.0366868
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0367506, upper bound: 0.0366767
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0381104, upper bound: 0.0381217
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0381656, upper bound: 0.0380478
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0378432, upper bound: 0.0379299
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0378924, upper bound: 0.0378766
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0378292, upper bound: 0.0379210
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0378763, upper bound: 0.0378749
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0363179, upper bound: 0.0363676
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0363179, upper bound: 0.0363676
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0376316, upper bound: 0.0375891
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0375796, upper bound: 0.0376392
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0359367, upper bound: 0.0359979
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0359373, upper bound: 0.0359967
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0373801, upper bound: 0.0373657
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 8, lower bound: -0.0374260, upper bound: 0.0373211

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171221, 0.0171283
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106468, 0.0106762
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431296, 0.0430856
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204484, 0.0204153
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326379, 0.0327095
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351617, upper bound: 0.0354165
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0353504, upper bound: 0.0352649
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171627, 0.0171203
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108182, 0.0106388
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430736, 0.0433731
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204063, 0.0206357
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331090, 0.0326183
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351759, upper bound: 0.0352410
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351741, upper bound: 0.0352440
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171545, 0.0171168
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107652, 0.0106221
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430485, 0.0433150
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203874, 0.0205848
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330080, 0.0325775
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352500, upper bound: 0.0351631
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352456, upper bound: 0.0351633
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171951, 0.0171069
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109362, 0.0105760
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429795, 0.0436022
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203355, 0.0208062
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334729, 0.0324650
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0772403
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352813, upper bound: 0.0352418
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0353265, upper bound: 0.0352117
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171561, 0.0171396
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106066, 0.0105284
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431033, 0.0432192
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204015, 0.0204890
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0327954, 0.0326060
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0771356
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0372655, upper bound: 0.0373699
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0374276, upper bound: 0.0372109
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171568, 0.0171407
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106096, 0.0105336
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431111, 0.0432239
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204073, 0.0204925
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328030, 0.0326187
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0771720
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367830, upper bound: 0.0368108
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367830, upper bound: 0.0368108
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171556, 0.0171477
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107496, 0.0107119
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432143, 0.0432697
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205071, 0.0205488
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0329494, 0.0328587
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0375174, upper bound: 0.0375536
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0375498, upper bound: 0.0375200
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171635, 0.0171400
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107866, 0.0106757
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431600, 0.0433250
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204663, 0.0205904
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330396, 0.0327703
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364846, upper bound: 0.0364599
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364846, upper bound: 0.0364599
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171406, 0.0171420
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107204, 0.0107273
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432289, 0.0432186
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205308, 0.0205230
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328639, 0.0328807
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367280, upper bound: 0.0370285
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367266, upper bound: 0.0370335
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171730, 0.0171294
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108388, 0.0106684
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431406, 0.0434473
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204644, 0.0206931
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332273, 0.0327368
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369221, upper bound: 0.0367459
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369508, upper bound: 0.0367356
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171707, 0.0171607
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108267, 0.0107799
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433602, 0.0434303
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206275, 0.0206802
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331984, 0.0330843
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362411, upper bound: 0.0362517
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362411, upper bound: 0.0362517
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171721, 0.0171609
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108346, 0.0107808
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433616, 0.0434410
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0206285, 0.0206884
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332170, 0.0330865
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0368366, upper bound: 0.0370277
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370109, upper bound: 0.0368838
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171626, 0.0171437
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108185, 0.0107348
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432407, 0.0433726
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205392, 0.0206356
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331090, 0.0328988
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365573, upper bound: 0.0368428
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367258, upper bound: 0.0367130
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171950, 0.0171304
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109366, 0.0106726
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431475, 0.0436017
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204692, 0.0208060
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334734, 0.0327470
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367461, upper bound: 0.0366922
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0368639, upper bound: 0.0364924
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171077, 0.0171099
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104243, 0.0104348
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428934, 0.0428776
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202517, 0.0202399
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0322453, 0.0322710
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0761453, 0.0762189
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369096, upper bound: 0.0370363
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369096, upper bound: 0.0370363
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171262, 0.0171087
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105107, 0.0104292
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428849, 0.0430068
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202454, 0.0203342
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324568, 0.0322573
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0767702, 0.0761796
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0312446, upper bound: 0.0312051
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0312446, upper bound: 0.0312051
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170808, 0.0170906
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104058, 0.0104517
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427320, 0.0426633
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201244, 0.0200728
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319076, 0.0320196
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0749934, 0.0753142
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365397, upper bound: 0.0369449
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365397, upper bound: 0.0369449
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170812, 0.0170916
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104074, 0.0104560
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427384, 0.0426657
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201292, 0.0200745
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319115, 0.0320299
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0750045, 0.0753438
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369434, upper bound: 0.0370854
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369063, upper bound: 0.0371479
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171252, 0.0171214
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106475, 0.0106295
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430324, 0.0430592
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203734, 0.0203936
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326141, 0.0325704
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0772554
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360233, upper bound: 0.0362720
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360091, upper bound: 0.0362841
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171320, 0.0171214
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106765, 0.0106296
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430324, 0.0431063
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203734, 0.0204304
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326945, 0.0325704
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0772555
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360324, upper bound: 0.0362205
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359946, upper bound: 0.0362927
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171761, 0.0171331
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108531, 0.0106849
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431660, 0.0434688
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204831, 0.0207093
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332623, 0.0327771
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370690, upper bound: 0.0369127
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370126, upper bound: 0.0369424
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171744, 0.0171342
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108455, 0.0106904
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431743, 0.0434574
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204893, 0.0207007
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332437, 0.0327906
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366859, upper bound: 0.0366206
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367223, upper bound: 0.0366046
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171930, 0.0171309
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109264, 0.0106747
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431506, 0.0435875
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204715, 0.0207952
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334489, 0.0327522
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355062, upper bound: 0.0353131
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355062, upper bound: 0.0353131
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171914, 0.0171322
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109188, 0.0106808
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431598, 0.0435761
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204784, 0.0207866
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334304, 0.0327670
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370216, upper bound: 0.0368412
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0370193, upper bound: 0.0368601
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171909, 0.0171289
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109240, 0.0106752
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431394, 0.0435750
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204679, 0.0207888
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334389, 0.0327445
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351339, upper bound: 0.0349324
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351313, upper bound: 0.0349431
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171977, 0.0171292
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109555, 0.0106766
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431415, 0.0436211
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204695, 0.0208256
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0335198, 0.0327481
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354504, upper bound: 0.0352450
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354401, upper bound: 0.0352660
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170950, 0.0171279
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103807, 0.0105370
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430522, 0.0428182
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203625, 0.0201865
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0321512, 0.0325326
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0757238, 0.0768165
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339716, upper bound: 0.0340478
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339716, upper bound: 0.0340478
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171275, 0.0171163
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104829, 0.0104827
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429708, 0.0430454
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203012, 0.0203544
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325179, 0.0323999
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0767135, 0.0764364
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344503, upper bound: 0.0341979
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344458, upper bound: 0.0342019
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170947, 0.0171171
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102954, 0.0104032
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428853, 0.0427241
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202279, 0.0201069
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319792, 0.0322423
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0747767, 0.0754871
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347537, upper bound: 0.0347459
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347537, upper bound: 0.0347459
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170954, 0.0171098
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102985, 0.0103686
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428335, 0.0427287
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201890, 0.0201104
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319868, 0.0321580
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0747983, 0.0752455
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350114, upper bound: 0.0349190
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350466, upper bound: 0.0348958
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170870, 0.0171153
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102596, 0.0103946
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428725, 0.0426702
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202182, 0.0200661
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0318918, 0.0322214
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0744830, 0.0754273
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341610, upper bound: 0.0344796
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343552, upper bound: 0.0343083
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170937, 0.0171151
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102905, 0.0103936
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428709, 0.0427168
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202170, 0.0201014
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319673, 0.0322188
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0747424, 0.0754199
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0346000, upper bound: 0.0347366
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0346000, upper bound: 0.0347366
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170944, 0.0171139
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102932, 0.0103870
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428620, 0.0427215
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202107, 0.0201050
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319756, 0.0322045
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0747592, 0.0754149
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360597, upper bound: 0.0359754
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360581, upper bound: 0.0359776
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170959, 0.0171141
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103008, 0.0103879
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428633, 0.0427321
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202117, 0.0201130
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319923, 0.0322067
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0748143, 0.0754212
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363790, upper bound: 0.0363458
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363785, upper bound: 0.0363474
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171403, 0.0171093
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105291, 0.0103870
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429008, 0.0431206
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202477, 0.0204228
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326627, 0.0322856
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0772265, 0.0760693
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359352, upper bound: 0.0359439
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359727, upper bound: 0.0359026
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171389, 0.0171108
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105223, 0.0103941
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429114, 0.0431104
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202557, 0.0204152
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326462, 0.0323029
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0771792, 0.0761190
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359094, upper bound: 0.0359345
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359576, upper bound: 0.0358828
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171685, 0.0171402
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106654, 0.0105346
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431251, 0.0433273
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204154, 0.0205746
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0329916, 0.0326422
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0770747
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362042, upper bound: 0.0362774
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362453, upper bound: 0.0362380
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171675, 0.0171405
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106606, 0.0105362
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431275, 0.0433201
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204173, 0.0205692
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0329799, 0.0326462
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0770861
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365100, upper bound: 0.0365335
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365097, upper bound: 0.0365336
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171692, 0.0171238
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108200, 0.0106379
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430594, 0.0433792
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203895, 0.0206312
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331006, 0.0325816
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0772186
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359739, upper bound: 0.0359542
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0359178, upper bound: 0.0359879
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171714, 0.0171214
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108303, 0.0106269
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430428, 0.0433946
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203771, 0.0206428
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331257, 0.0325546
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0771412
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0281498, upper bound: 0.0280363
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0281498, upper bound: 0.0280363
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170956, 0.0171161
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103131, 0.0104092
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429363, 0.0427925
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202737, 0.0201655
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0321012, 0.0323357
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0755311, 0.0762027
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355080, upper bound: 0.0356359
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356826, upper bound: 0.0354806
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171365, 0.0171068
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105037, 0.0103654
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428707, 0.0430841
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202243, 0.0203921
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325962, 0.0322287
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0770422, 0.0758963
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363163, upper bound: 0.0363012
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363759, upper bound: 0.0362330
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170977, 0.0171133
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102848, 0.0103575
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428332, 0.0427243
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201948, 0.0201129
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319904, 0.0321678
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0748963, 0.0754047
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0368201, upper bound: 0.0372252
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371882, upper bound: 0.0370104
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170993, 0.0171108
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102925, 0.0103455
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428152, 0.0427359
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201812, 0.0201216
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0320092, 0.0321385
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0749502, 0.0753205
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369835, upper bound: 0.0368364
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369835, upper bound: 0.0368364
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170746, 0.0170958
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102107, 0.0103079
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427172, 0.0425716
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201018, 0.0199923
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0317538, 0.0319911
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0742923, 0.0749720
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0374946, upper bound: 0.0375630
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0374946, upper bound: 0.0375630
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170812, 0.0170881
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102416, 0.0102718
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0426631, 0.0426178
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0200611, 0.0200271
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0318292, 0.0319030
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0745083, 0.0747196
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361548, upper bound: 0.0361359
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361801, upper bound: 0.0361264
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170943, 0.0171114
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103159, 0.0103998
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429005, 0.0427749
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202548, 0.0201603
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0320740, 0.0322786
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0756422, 0.0762285
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364666, upper bound: 0.0365457
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364666, upper bound: 0.0365457
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171128, 0.0171090
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103965, 0.0103884
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428835, 0.0429048
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202420, 0.0202551
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0322886, 0.0322510
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0762480, 0.0761492
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0318558, upper bound: 0.0318261
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0318558, upper bound: 0.0318261
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171108, 0.0171327
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105373, 0.0106379
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431227, 0.0429720
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204410, 0.0203277
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324632, 0.0327087
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0769270, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362490, upper bound: 0.0362529
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361994, upper bound: 0.0363047
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171177, 0.0171327
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105683, 0.0106379
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431226, 0.0430197
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204409, 0.0203648
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325442, 0.0327085
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0771831, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356318, upper bound: 0.0356763
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356318, upper bound: 0.0356763
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170902, 0.0170998
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102877, 0.0103314
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427734, 0.0427079
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201487, 0.0200994
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319758, 0.0320826
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0748949, 0.0752008
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364523, upper bound: 0.0366698
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367248, upper bound: 0.0364898
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170893, 0.0171006
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0102835, 0.0103352
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427791, 0.0427017
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201530, 0.0200947
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319656, 0.0320919
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0748658, 0.0752275
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364472, upper bound: 0.0367179
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366875, upper bound: 0.0365200
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170953, 0.0171099
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104930, 0.0105616
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429594, 0.0428568
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203135, 0.0202362
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0322878, 0.0324499
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0764015, 0.0768787
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356070, upper bound: 0.0357201
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356522, upper bound: 0.0356504
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171361, 0.0170983
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106524, 0.0105073
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428780, 0.0431452
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202523, 0.0204591
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0327649, 0.0323173
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0764989
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355895, upper bound: 0.0355863
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355895, upper bound: 0.0355863
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171031, 0.0170970
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104657, 0.0104441
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427872, 0.0428265
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201752, 0.0202031
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0321954, 0.0321306
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0757985, 0.0756377
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360212, upper bound: 0.0360549
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360212, upper bound: 0.0360549
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171055, 0.0170930
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104768, 0.0104252
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427590, 0.0428431
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201540, 0.0202157
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0322226, 0.0320845
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0758764, 0.0755057
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363214, upper bound: 0.0363823
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365142, upper bound: 0.0360371
time: 0.87 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0351617, upper bound: 0.0354165
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0353504, upper bound: 0.0352649
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0351759, upper bound: 0.0352410
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0351741, upper bound: 0.0352440
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0352500, upper bound: 0.0351631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0352456, upper bound: 0.0351633
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0352813, upper bound: 0.0352418
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0353265, upper bound: 0.0352117
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0372655, upper bound: 0.0373699
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0374276, upper bound: 0.0372109
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0367830, upper bound: 0.0368108
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0367830, upper bound: 0.0368108
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0375174, upper bound: 0.0375536
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0375498, upper bound: 0.0375200
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0364846, upper bound: 0.0364599
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0364846, upper bound: 0.0364599
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0367280, upper bound: 0.0370285
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0367266, upper bound: 0.0370335
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0369221, upper bound: 0.0367459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0369508, upper bound: 0.0367356
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0362411, upper bound: 0.0362517
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0362411, upper bound: 0.0362517
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0368366, upper bound: 0.0370277
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0370109, upper bound: 0.0368838
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0365573, upper bound: 0.0368428
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0367258, upper bound: 0.0367130
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0367461, upper bound: 0.0366922
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0368639, upper bound: 0.0364924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0369096, upper bound: 0.0370363
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0369096, upper bound: 0.0370363
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0312446, upper bound: 0.0312051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0312446, upper bound: 0.0312051
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0365397, upper bound: 0.0369449
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0365397, upper bound: 0.0369449
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0369434, upper bound: 0.0370854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0369063, upper bound: 0.0371479
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0360233, upper bound: 0.0362720
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0360091, upper bound: 0.0362841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0360324, upper bound: 0.0362205
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0359946, upper bound: 0.0362927
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0370690, upper bound: 0.0369127
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0370126, upper bound: 0.0369424
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0366859, upper bound: 0.0366206
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0367223, upper bound: 0.0366046
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0355062, upper bound: 0.0353131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0355062, upper bound: 0.0353131
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0370216, upper bound: 0.0368412
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0370193, upper bound: 0.0368601
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0351339, upper bound: 0.0349324
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0351313, upper bound: 0.0349431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0354504, upper bound: 0.0352450
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0354401, upper bound: 0.0352660
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0339716, upper bound: 0.0340478
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0339716, upper bound: 0.0340478
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0344503, upper bound: 0.0341979
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0344458, upper bound: 0.0342019
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0347537, upper bound: 0.0347459
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0347537, upper bound: 0.0347459
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0350114, upper bound: 0.0349190
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0350466, upper bound: 0.0348958
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0341610, upper bound: 0.0344796
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0343552, upper bound: 0.0343083
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0346000, upper bound: 0.0347366
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0346000, upper bound: 0.0347366
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0360597, upper bound: 0.0359754
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0360581, upper bound: 0.0359776
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0363790, upper bound: 0.0363458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0363785, upper bound: 0.0363474
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0359352, upper bound: 0.0359439
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0359727, upper bound: 0.0359026
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0359094, upper bound: 0.0359345
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0359576, upper bound: 0.0358828
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0362042, upper bound: 0.0362774
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0362453, upper bound: 0.0362380
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0365100, upper bound: 0.0365335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0365097, upper bound: 0.0365336
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0359739, upper bound: 0.0359542
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0359178, upper bound: 0.0359879
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0281498, upper bound: 0.0280363
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0281498, upper bound: 0.0280363
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0355080, upper bound: 0.0356359
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0356826, upper bound: 0.0354806
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0363163, upper bound: 0.0363012
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0363759, upper bound: 0.0362330
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0368201, upper bound: 0.0372252
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0371882, upper bound: 0.0370104
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0369835, upper bound: 0.0368364
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0369835, upper bound: 0.0368364
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0374946, upper bound: 0.0375630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0374946, upper bound: 0.0375630
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0361548, upper bound: 0.0361359
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0361801, upper bound: 0.0361264
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0364666, upper bound: 0.0365457
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0364666, upper bound: 0.0365457
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0318558, upper bound: 0.0318261
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0318558, upper bound: 0.0318261
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0362490, upper bound: 0.0362529
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0361994, upper bound: 0.0363047
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0356318, upper bound: 0.0356763
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0356318, upper bound: 0.0356763
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0364523, upper bound: 0.0366698
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0367248, upper bound: 0.0364898
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0364472, upper bound: 0.0367179
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0366875, upper bound: 0.0365200
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0356070, upper bound: 0.0357201
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0356522, upper bound: 0.0356504
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0355895, upper bound: 0.0355863
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0355895, upper bound: 0.0355863
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0360212, upper bound: 0.0360549
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0360212, upper bound: 0.0360549
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0363214, upper bound: 0.0363823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.44
Output dim: 8, lower bound: -0.0365142, upper bound: 0.0360371

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170801, 0.0170908
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104371, 0.0104869
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428004, 0.0427259
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201879, 0.0201319
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0320238, 0.0321452
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0755811, 0.0759290
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271093, upper bound: 0.0271934
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271093, upper bound: 0.0271934
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170808, 0.0170864
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104401, 0.0104665
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427698, 0.0427303
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201649, 0.0201352
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0320310, 0.0320954
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0756018, 0.0757861
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348722, upper bound: 0.0347700
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348693, upper bound: 0.0347700
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171273, 0.0170851
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106402, 0.0104614
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428079, 0.0431066
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202065, 0.0204353
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326746, 0.0321853
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0764389
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349704, upper bound: 0.0350563
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349596, upper bound: 0.0350643
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171274, 0.0170874
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106408, 0.0104721
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428239, 0.0431074
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202185, 0.0204359
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326760, 0.0322114
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0765137
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335305, upper bound: 0.0336472
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335305, upper bound: 0.0336472
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171191, 0.0170815
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105872, 0.0104446
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427828, 0.0430484
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201876, 0.0203843
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325737, 0.0321444
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0763218
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271205, upper bound: 0.0271204
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271205, upper bound: 0.0271204
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171192, 0.0170825
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105877, 0.0104494
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427900, 0.0430492
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201930, 0.0203849
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325750, 0.0321562
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0763555
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351672, upper bound: 0.0350478
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351183, upper bound: 0.0350747
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171586, 0.0170779
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107623, 0.0104519
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427285, 0.0432897
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201406, 0.0205638
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0329816, 0.0320673
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0758679
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348105, upper bound: 0.0348029
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348195, upper bound: 0.0347584
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171665, 0.0170704
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107992, 0.0104164
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0426753, 0.0433450
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201006, 0.0206054
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330717, 0.0319805
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0756193
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352576, upper bound: 0.0350953
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352285, upper bound: 0.0351340
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171237, 0.0171206
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104992, 0.0104838
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429672, 0.0429895
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203041, 0.0203212
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324287, 0.0323918
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0766895, 0.0765803
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357862, upper bound: 0.0359348
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357862, upper bound: 0.0359348
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171561, 0.0171072
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106066, 0.0104212
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428736, 0.0432192
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202336, 0.0204890
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0327954, 0.0322391
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0761429
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360064, upper bound: 0.0357206
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360064, upper bound: 0.0357206
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171500, 0.0171336
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105768, 0.0104990
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430627, 0.0431781
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203680, 0.0204552
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0327219, 0.0325333
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0768864
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363594, upper bound: 0.0364125
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363865, upper bound: 0.0363907
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171568, 0.0171340
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106096, 0.0105007
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430653, 0.0432239
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203700, 0.0204925
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328030, 0.0325376
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0768986
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363886, upper bound: 0.0365334
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365077, upper bound: 0.0364308
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171371, 0.0171292
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106625, 0.0106249
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430875, 0.0431428
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204107, 0.0204524
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0327286, 0.0326381
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371782, upper bound: 0.0372830
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0372385, upper bound: 0.0372259
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171556, 0.0171292
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107496, 0.0106248
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430874, 0.0432697
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204106, 0.0205488
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0329494, 0.0326380
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0372099, upper bound: 0.0372450
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0372733, upper bound: 0.0371979
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171567, 0.0171326
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107562, 0.0106426
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431088, 0.0432781
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204262, 0.0205534
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0329580, 0.0326819
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295505, upper bound: 0.0295087
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295505, upper bound: 0.0295087
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171635, 0.0171332
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107866, 0.0106454
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431131, 0.0433250
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204294, 0.0205904
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330396, 0.0326888
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360933, upper bound: 0.0360570
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0360931, upper bound: 0.0360591
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171051, 0.0171069
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105419, 0.0105502
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429636, 0.0429512
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203313, 0.0203219
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324282, 0.0324484
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0771724, 0.0772305
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0308453, upper bound: 0.0309295
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0308453, upper bound: 0.0309295
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171054, 0.0171081
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105433, 0.0105563
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429727, 0.0429533
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203381, 0.0203235
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324316, 0.0324633
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0771821, 0.0772729
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366451, upper bound: 0.0369089
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366362, upper bound: 0.0369629
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171364, 0.0170982
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106592, 0.0105192
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428717, 0.0431379
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202539, 0.0204487
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0327206, 0.0322995
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0765417
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356120, upper bound: 0.0353781
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356120, upper bound: 0.0353781
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171450, 0.0170929
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106992, 0.0104940
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428340, 0.0431978
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202255, 0.0204938
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328183, 0.0322381
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0763657
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0368996, upper bound: 0.0366376
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0368549, upper bound: 0.0366651
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171639, 0.0171529
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107955, 0.0107441
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433071, 0.0433840
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205855, 0.0206434
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331163, 0.0329910
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361804, upper bound: 0.0361343
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0361315, upper bound: 0.0361929
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171707, 0.0171539
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108267, 0.0107487
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0433140, 0.0434303
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205907, 0.0206802
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0331984, 0.0330022
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0358437, upper bound: 0.0358633
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0358366, upper bound: 0.0358709
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171397, 0.0171407
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107162, 0.0107204
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0432191, 0.0432123
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0205230, 0.0205182
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328536, 0.0328637
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352096, upper bound: 0.0354261
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352096, upper bound: 0.0354261
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171721, 0.0171284
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108346, 0.0106628
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431329, 0.0434410
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204582, 0.0206884
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0332170, 0.0327232
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365936, upper bound: 0.0364685
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365930, upper bound: 0.0364685
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171207, 0.0171025
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106041, 0.0105287
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428877, 0.0430132
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202609, 0.0203513
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324852, 0.0322850
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0769640, 0.0764210
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347263, upper bound: 0.0349400
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347275, upper bound: 0.0349400
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171211, 0.0171017
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106059, 0.0105248
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428818, 0.0430160
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202565, 0.0203534
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0324898, 0.0322754
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0769771, 0.0763935
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363357, upper bound: 0.0362664
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363357, upper bound: 0.0362664
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171535, 0.0170911
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107260, 0.0104753
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428077, 0.0432413
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202008, 0.0205208
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328570, 0.0321547
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0760476
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351585, upper bound: 0.0350426
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351585, upper bound: 0.0350426
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171539, 0.0170884
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107279, 0.0104626
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427886, 0.0432441
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201864, 0.0205229
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328616, 0.0321236
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0759585
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364827, upper bound: 0.0360098
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364827, upper bound: 0.0360098
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171062, 0.0171082
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104166, 0.0104260
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428809, 0.0428669
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202419, 0.0202314
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0322273, 0.0322502
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0760890, 0.0761546
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352420, upper bound: 0.0353901
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352420, upper bound: 0.0353901
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171077, 0.0171084
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104243, 0.0104271
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428827, 0.0428776
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202432, 0.0202399
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0322453, 0.0322530
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0761453, 0.0761627
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350710, upper bound: 0.0351289
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350811, upper bound: 0.0351284
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170794, 0.0170893
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103985, 0.0104448
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427220, 0.0426526
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201169, 0.0200647
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0318898, 0.0320029
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0749387, 0.0752627
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347341, upper bound: 0.0350467
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347346, upper bound: 0.0350466
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170808, 0.0170892
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0104058, 0.0104444
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0427213, 0.0426633
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201164, 0.0200728
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0319076, 0.0320017
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0749934, 0.0752595
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0364531, upper bound: 0.0368363
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363992, upper bound: 0.0368851
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170462, 0.0170566
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0100450, 0.0100935
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0423301, 0.0422574
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0197991, 0.0197444
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0311916, 0.0313101
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0721192, 0.0724586
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351260, upper bound: 0.0351418
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351434, upper bound: 0.0351418
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170462, 0.0170609
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0100450, 0.0101137
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0423602, 0.0422574
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0198217, 0.0197444
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0311917, 0.0313592
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0721192, 0.0725992
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356934, upper bound: 0.0359674
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356934, upper bound: 0.0359674
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171216, 0.0171170
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106309, 0.0106091
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430018, 0.0430344
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203504, 0.0203749
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325736, 0.0325206
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0772646, 0.0771125
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291316, upper bound: 0.0291492
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291316, upper bound: 0.0291492
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171208, 0.0171175
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106270, 0.0106116
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0430056, 0.0430286
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0203533, 0.0203706
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0325643, 0.0325267
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0772378, 0.0771302
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0353762, upper bound: 0.0356242
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0353777, upper bound: 0.0356242
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170973, 0.0170841
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103597, 0.0102986
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0426489, 0.0427413
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0200545, 0.0201253
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0320385, 0.0318849
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0750393, 0.0745694
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355850, upper bound: 0.0359357
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357588, upper bound: 0.0358271
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0170947, 0.0170865
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0103476, 0.0103100
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0426659, 0.0427233
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0200672, 0.0201117
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0320091, 0.0319126
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0749550, 0.0746487
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291687, upper bound: 0.0291895
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291687, upper bound: 0.0291895
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171401, 0.0170967
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105358, 0.0103723
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428005, 0.0431066
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201815, 0.0204079
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326082, 0.0321192
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0771521, 0.0757792
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366340, upper bound: 0.0365054
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366690, upper bound: 0.0364689
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171397, 0.0170973
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0105338, 0.0103752
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428050, 0.0431036
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0201848, 0.0204057
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0326034, 0.0321264
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0771382, 0.0757999
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365788, upper bound: 0.0366541
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367257, upper bound: 0.0364742
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171378, 0.0171034
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0106655, 0.0105425
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0429072, 0.0431473
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202801, 0.0204558
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0327359, 0.0323567
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0767017
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362672, upper bound: 0.0362346
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0362638, upper bound: 0.0362346
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171460, 0.0170977
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107035, 0.0105157
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428671, 0.0432043
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202500, 0.0204986
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0328288, 0.0322913
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0765143
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363287, upper bound: 0.0361968
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0363287, upper bound: 0.0362002
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171863, 0.0171238
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108948, 0.0106425
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431027, 0.0435414
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204345, 0.0207585
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0333684, 0.0326667
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350945, upper bound: 0.0349126
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351244, upper bound: 0.0348881
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171930, 0.0171241
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109264, 0.0106437
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431044, 0.0435875
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204358, 0.0207952
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334489, 0.0326695
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351552, upper bound: 0.0350462
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352201, upper bound: 0.0348616
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171876, 0.0171277
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109010, 0.0106599
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431285, 0.0435493
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204549, 0.0207665
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0333868, 0.0327160
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365955, upper bound: 0.0364377
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0366221, upper bound: 0.0364094
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171870, 0.0171284
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0108979, 0.0106628
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431329, 0.0435448
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204582, 0.0207631
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0333794, 0.0327232
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369642, upper bound: 0.0367733
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369265, upper bound: 0.0367951
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171567, 0.0170939
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107512, 0.0104988
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428753, 0.0433163
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202693, 0.0205942
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330172, 0.0323143
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0768108
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351024, upper bound: 0.0348885
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350893, upper bound: 0.0348972
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171559, 0.0170945
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0107477, 0.0105016
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0428795, 0.0433110
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0202725, 0.0205903
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0330086, 0.0323211
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0768303
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347140, upper bound: 0.0345540
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347667, upper bound: 0.0345290
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171940, 0.0171248
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109379, 0.0106559
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431105, 0.0435946
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204462, 0.0208057
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334767, 0.0326976
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350523, upper bound: 0.0348126
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0350490, upper bound: 0.0348214
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0133591, 0.0043988, -0.0133591, 0.0043988, -0.0171933, 0.0171254
1: -0.0030181, 0.0080630, -0.0030181, 0.0080630, -0.0109349, 0.0106587
2: 0.0042648, 0.0481298, 0.0042648, 0.0481298, -0.0431147, 0.0435901
3: -0.0074200, 0.0138134, -0.0074200, 0.0138134, -0.0204493, 0.0208023
4: -0.0112238, 0.0262155, -0.0112238, 0.0262155, -0.0374393, 0.0374393
5: 0.0002488, 0.0122308, 0.0002488, 0.0122308, -0.0119821, 0.0119821
6: -0.0003501, 0.0127322, -0.0003501, 0.0127322, -0.0130823, 0.0130823
7: -0.0376248, 0.0004461, -0.0376248, 0.0004461, -0.0334693, 0.0327043
8: 0.9477017, 1.0250691, 0.9477017, 1.0250691, -0.0773674, 0.0773674
9: -0.0109327, 0.0099672, -0.0109327, 0.0099672, -0.0208999, 0.0208999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 3.34 + 597.00 = 600.34 seconds
