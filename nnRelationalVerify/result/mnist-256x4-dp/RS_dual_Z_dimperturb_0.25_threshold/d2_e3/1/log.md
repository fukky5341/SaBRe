## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00018056


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001488, 0.0001488)
1: (0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002883, 0.0002883)
2: (0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0023252, 0.0023252)
3: (-0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0002077, 0.0002077)
4: (0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0010076, 0.0010076)
5: (-0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001504, 0.0001504)
6: (0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002759, 0.0002759)
7: (-0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0018239, 0.0018239)
8: (0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005714, 0.0005714)
9: (-0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0011405, 0.0011405)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.47 = 2.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0002120, upper bound: 0.0002119

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002042, upper bound: 0.0001926
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001925, upper bound: 0.0002042
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 6, lower bound: -0.0002042, upper bound: 0.0001926
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 6, lower bound: -0.0001925, upper bound: 0.0002042

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001469, 0.0001428
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002845, 0.0002765
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0022306, 0.0022947
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0002050, 0.0001992
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009944, 0.0009666
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001443, 0.0001484
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002723, 0.0002646
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0018000, 0.0017497
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005639, 0.0005482
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010941, 0.0011255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002003, upper bound: 0.0001802
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001957, upper bound: 0.0001883
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001428, 0.0001469
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002765, 0.0002845
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0022947, 0.0022306
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001992, 0.0002050
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009666, 0.0009944
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001484, 0.0001443
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002646, 0.0002723
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0017497, 0.0018000
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005482, 0.0005639
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0011255, 0.0010941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001883, upper bound: 0.0001957
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001801, upper bound: 0.0002003
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 6, lower bound: -0.0002003, upper bound: 0.0001802
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 6, lower bound: -0.0001957, upper bound: 0.0001883
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 6, lower bound: -0.0001883, upper bound: 0.0001957
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 6, lower bound: -0.0001801, upper bound: 0.0002003

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001378, 0.0001323
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002669, 0.0002562
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0020668, 0.0021529
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001923, 0.0001846
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009330, 0.0008956
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001337, 0.0001393
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002554, 0.0002452
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016888, 0.0016212
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005291, 0.0005079
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010137, 0.0010560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001981, upper bound: 0.0001735
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001898, upper bound: 0.0001779
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001364, 0.0001338
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002642, 0.0002592
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0020904, 0.0021309
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001903, 0.0001867
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009234, 0.0009058
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001352, 0.0001378
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002528, 0.0002480
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016715, 0.0016397
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005237, 0.0005137
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010253, 0.0010452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001934, upper bound: 0.0001789
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001852, upper bound: 0.0001858
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001338, 0.0001364
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002592, 0.0002642
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0021309, 0.0020904
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001867, 0.0001903
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009058, 0.0009234
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001378, 0.0001352
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002480, 0.0002528
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016397, 0.0016715
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005137, 0.0005237
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010452, 0.0010253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001858, upper bound: 0.0001852
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001788, upper bound: 0.0001934
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001323, 0.0001378
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002562, 0.0002669
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0021529, 0.0020668
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001846, 0.0001923
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0008956, 0.0009330
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001393, 0.0001337
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002452, 0.0002554
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016212, 0.0016888
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005079, 0.0005291
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010560, 0.0010137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001778, upper bound: 0.0001898
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001734, upper bound: 0.0001981
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 6, lower bound: -0.0001981, upper bound: 0.0001735
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 6, lower bound: -0.0001898, upper bound: 0.0001779
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 6, lower bound: -0.0001934, upper bound: 0.0001789
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 6, lower bound: -0.0001852, upper bound: 0.0001858
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 6, lower bound: -0.0001858, upper bound: 0.0001852
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 6, lower bound: -0.0001788, upper bound: 0.0001934
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 6, lower bound: -0.0001778, upper bound: 0.0001898
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 6, lower bound: -0.0001734, upper bound: 0.0001981

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001379, 0.0001317
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002671, 0.0002551
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0020572, 0.0021547
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001924, 0.0001837
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009337, 0.0008915
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001331, 0.0001394
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002556, 0.0002441
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016901, 0.0016137
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005295, 0.0005056
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010090, 0.0010568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001881, upper bound: 0.0001549
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001768, upper bound: 0.0001616
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001373, 0.0001324
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002660, 0.0002565
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0020685, 0.0021455
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001916, 0.0001847
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009297, 0.0008964
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001338, 0.0001388
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002546, 0.0002454
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016830, 0.0016226
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005273, 0.0005083
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010146, 0.0010524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001765, upper bound: 0.0001580
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001733, upper bound: 0.0001669
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001365, 0.0001333
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002644, 0.0002582
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0020825, 0.0021326
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001905, 0.0001860
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009242, 0.0009024
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001347, 0.0001380
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002530, 0.0002471
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016729, 0.0016335
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005241, 0.0005118
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010214, 0.0010460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001821, upper bound: 0.0001603
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001733, upper bound: 0.0001675
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001359, 0.0001339
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002632, 0.0002594
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0020921, 0.0021233
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001896, 0.0001869
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009201, 0.0009066
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001353, 0.0001374
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002519, 0.0002482
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016656, 0.0016411
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005218, 0.0005141
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010262, 0.0010415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001717, upper bound: 0.0001637
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001680, upper bound: 0.0001754
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001339, 0.0001359
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002594, 0.0002632
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0021233, 0.0020921
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001869, 0.0001896
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009066, 0.0009201
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001374, 0.0001353
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002482, 0.0002519
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016411, 0.0016656
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005141, 0.0005218
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010415, 0.0010262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001754, upper bound: 0.0001680
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001637, upper bound: 0.0001716
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001333, 0.0001365
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002582, 0.0002644
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0021326, 0.0020825
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001860, 0.0001905
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0009024, 0.0009242
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001380, 0.0001347
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002471, 0.0002530
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016335, 0.0016729
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005118, 0.0005241
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010460, 0.0010214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001675, upper bound: 0.0001732
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001603, upper bound: 0.0001821
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001324, 0.0001373
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002565, 0.0002660
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0021455, 0.0020685
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001847, 0.0001916
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0008964, 0.0009297
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001388, 0.0001338
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002454, 0.0002546
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016226, 0.0016830
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005083, 0.0005273
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010524, 0.0010146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001669, upper bound: 0.0001733
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001580, upper bound: 0.0001766
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001317, 0.0001379
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002551, 0.0002671
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0021547, 0.0020572
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001837, 0.0001924
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0008915, 0.0009337
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001394, 0.0001331
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002441, 0.0002556
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0016137, 0.0016901
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0005056, 0.0005295
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0010568, 0.0010090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001616, upper bound: 0.0001769
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001549, upper bound: 0.0001880
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001881, upper bound: 0.0001549
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001768, upper bound: 0.0001616
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001765, upper bound: 0.0001580
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001733, upper bound: 0.0001669
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001821, upper bound: 0.0001603
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001733, upper bound: 0.0001675
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001717, upper bound: 0.0001637
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001680, upper bound: 0.0001754
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001754, upper bound: 0.0001680
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001637, upper bound: 0.0001716
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001675, upper bound: 0.0001732
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001603, upper bound: 0.0001821
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001669, upper bound: 0.0001733
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001580, upper bound: 0.0001766
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001616, upper bound: 0.0001769
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 6, lower bound: -0.0001549, upper bound: 0.0001880

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001268, 0.0001162
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002456, 0.0002251
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0018157, 0.0019806
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001769, 0.0001622
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0008583, 0.0007868
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001175, 0.0001281
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002350, 0.0002154
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0015537, 0.0014243
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0004868, 0.0004462
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0008906, 0.0009715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001635, upper bound: 0.0001450
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001635, upper bound: 0.0001450
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001252, 0.0001178
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002424, 0.0002282
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0018409, 0.0019552
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001746, 0.0001644
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0008473, 0.0007978
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001191, 0.0001265
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002320, 0.0002184
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0015337, 0.0014441
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0004805, 0.0004524
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0009030, 0.0009590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001582, upper bound: 0.0001480
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001582, upper bound: 0.0001482
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001178, 0.0001252
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002282, 0.0002424
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0019552, 0.0018409
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001644, 0.0001746
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0007978, 0.0008473
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001265, 0.0001191
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002184, 0.0002320
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0014441, 0.0015337
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0004524, 0.0004805
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0009590, 0.0009030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001482, upper bound: 0.0001582
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001480, upper bound: 0.0001582
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.0068570, 0.0071361, 0.0068570, 0.0071361, -0.0001162, 0.0001268
1: 0.0013636, 0.0019040, 0.0013636, 0.0019040, -0.0002251, 0.0002456
2: 0.0015701, 0.0059297, 0.0015701, 0.0059297, -0.0019806, 0.0018157
3: -0.0030422, -0.0026528, -0.0030422, -0.0026528, -0.0001622, 0.0001769
4: 0.0065761, 0.0084652, 0.0065761, 0.0084652, -0.0007868, 0.0008583
5: -0.0017834, -0.0015014, -0.0017834, -0.0015014, -0.0001281, 0.0001175
6: 0.9930068, 0.9935240, 0.9930068, 0.9935240, -0.0002154, 0.0002350
7: -0.0014790, 0.0019407, -0.0014790, 0.0019407, -0.0014243, 0.0015537
8: 0.0005250, 0.0015964, 0.0005250, 0.0015964, -0.0004462, 0.0004868
9: -0.0105152, -0.0083769, -0.0105152, -0.0083769, -0.0009715, 0.0008906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 237
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001450, upper bound: 0.0001635
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001450, upper bound: 0.0001635
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 6, lower bound: -0.0001635, upper bound: 0.0001450
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 6, lower bound: -0.0001635, upper bound: 0.0001450
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 6, lower bound: -0.0001582, upper bound: 0.0001480
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 6, lower bound: -0.0001582, upper bound: 0.0001482
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 6, lower bound: -0.0001482, upper bound: 0.0001582
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 6, lower bound: -0.0001480, upper bound: 0.0001582
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 6, lower bound: -0.0001450, upper bound: 0.0001635
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.59
Output dim: 6, lower bound: -0.0001450, upper bound: 0.0001635

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.86 + 47.57 = 50.43 seconds
