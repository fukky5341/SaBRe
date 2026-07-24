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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0006163, 0.0100718, -0.0006163, 0.0100718, -0.0101338, 0.0101338)
1: (-0.0035800, 0.0021775, -0.0035800, 0.0021775, -0.0056629, 0.0056629)
2: (0.0069218, 0.0168613, 0.0069218, 0.0168613, -0.0099395, 0.0099395)
3: (1.0059226, 1.0071489, 1.0059226, 1.0071489, -0.0012263, 0.0012263)
4: (-0.0043803, -0.0012098, -0.0043803, -0.0012098, -0.0031706, 0.0031706)
5: (0.0035104, 0.0163329, 0.0035104, 0.0163329, -0.0123861, 0.0123861)
6: (-0.0122887, -0.0025326, -0.0122887, -0.0025326, -0.0097561, 0.0097561)
7: (-0.0173318, -0.0102117, -0.0173318, -0.0102117, -0.0070577, 0.0070577)
8: (-0.0150475, -0.0075314, -0.0150475, -0.0075314, -0.0075160, 0.0075160)
9: (-0.0052566, 0.0032634, -0.0052566, 0.0032634, -0.0085200, 0.0085200)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.77 = 3.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0008342, upper bound: 0.0008341

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007602, upper bound: 0.0007602
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007602, upper bound: 0.0007602
time: 0.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 3, lower bound: -0.0007602, upper bound: 0.0007602
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 3, lower bound: -0.0007602, upper bound: 0.0007602

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0006163, 0.0100718, -0.0006163, 0.0100718, -0.0101282, 0.0101304
1: -0.0035800, 0.0021775, -0.0035800, 0.0021775, -0.0056602, 0.0056613
2: 0.0069218, 0.0168613, 0.0069218, 0.0168613, -0.0099395, 0.0099395
3: 1.0059226, 1.0071489, 1.0059226, 1.0071489, -0.0012263, 0.0012263
4: -0.0043803, -0.0012098, -0.0043803, -0.0012098, -0.0031706, 0.0031706
5: 0.0035104, 0.0163329, 0.0035104, 0.0163329, -0.0123816, 0.0123834
6: -0.0122887, -0.0025326, -0.0122887, -0.0025326, -0.0097561, 0.0097561
7: -0.0173318, -0.0102117, -0.0173318, -0.0102117, -0.0070573, 0.0070570
8: -0.0150475, -0.0075314, -0.0150475, -0.0075314, -0.0075160, 0.0075160
9: -0.0052566, 0.0032634, -0.0052566, 0.0032634, -0.0085200, 0.0085200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007239, upper bound: 0.0007250
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007250, upper bound: 0.0007239
time: 0.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0006163, 0.0100718, -0.0006163, 0.0100718, -0.0101304, 0.0101338
1: -0.0035800, 0.0021775, -0.0035800, 0.0021775, -0.0056613, 0.0056629
2: 0.0069218, 0.0168613, 0.0069218, 0.0168613, -0.0099395, 0.0099395
3: 1.0059226, 1.0071489, 1.0059226, 1.0071489, -0.0012263, 0.0012263
4: -0.0043803, -0.0012098, -0.0043803, -0.0012098, -0.0031706, 0.0031706
5: 0.0035104, 0.0163329, 0.0035104, 0.0163329, -0.0123834, 0.0123861
6: -0.0122887, -0.0025326, -0.0122887, -0.0025326, -0.0097561, 0.0097561
7: -0.0173318, -0.0102117, -0.0173318, -0.0102117, -0.0070577, 0.0070573
8: -0.0150475, -0.0075314, -0.0150475, -0.0075314, -0.0075160, 0.0075160
9: -0.0052566, 0.0032634, -0.0052566, 0.0032634, -0.0085200, 0.0085200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 189
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 189

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007239, upper bound: 0.0007250
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007249, upper bound: 0.0007239
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 3, lower bound: -0.0007239, upper bound: 0.0007250
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 3, lower bound: -0.0007250, upper bound: 0.0007239
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 3, lower bound: -0.0007239, upper bound: 0.0007250
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 3, lower bound: -0.0007249, upper bound: 0.0007239

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0006163, 0.0100718, -0.0006163, 0.0100718, -0.0100055, 0.0099899
1: -0.0035800, 0.0021775, -0.0035800, 0.0021775, -0.0056469, 0.0056393
2: 0.0069218, 0.0168613, 0.0069218, 0.0168613, -0.0099395, 0.0099395
3: 1.0059226, 1.0071489, 1.0059226, 1.0071489, -0.0012263, 0.0012263
4: -0.0043803, -0.0012098, -0.0043803, -0.0012098, -0.0031706, 0.0031706
5: 0.0035104, 0.0163329, 0.0035104, 0.0163329, -0.0122851, 0.0122729
6: -0.0122887, -0.0025326, -0.0122887, -0.0025326, -0.0097561, 0.0097561
7: -0.0173318, -0.0102117, -0.0173318, -0.0102117, -0.0070411, 0.0070429
8: -0.0150475, -0.0075314, -0.0150475, -0.0075314, -0.0075160, 0.0075160
9: -0.0052566, 0.0032634, -0.0052566, 0.0032634, -0.0085200, 0.0085200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006398, upper bound: 0.0006419
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006398, upper bound: 0.0006419
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0006163, 0.0100718, -0.0006163, 0.0100718, -0.0099877, 0.0100069
1: -0.0035800, 0.0021775, -0.0035800, 0.0021775, -0.0056382, 0.0056476
2: 0.0069218, 0.0168613, 0.0069218, 0.0168613, -0.0099395, 0.0099395
3: 1.0059226, 1.0071489, 1.0059226, 1.0071489, -0.0012263, 0.0012263
4: -0.0043803, -0.0012098, -0.0043803, -0.0012098, -0.0031706, 0.0031706
5: 0.0035104, 0.0163329, 0.0035104, 0.0163329, -0.0122711, 0.0122862
6: -0.0122887, -0.0025326, -0.0122887, -0.0025326, -0.0097561, 0.0097561
7: -0.0173318, -0.0102117, -0.0173318, -0.0102117, -0.0070431, 0.0070408
8: -0.0150475, -0.0075314, -0.0150475, -0.0075314, -0.0075160, 0.0075160
9: -0.0052566, 0.0032634, -0.0052566, 0.0032634, -0.0085200, 0.0085200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006419, upper bound: 0.0006399
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006419, upper bound: 0.0006398
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0006163, 0.0100718, -0.0006163, 0.0100718, -0.0100069, 0.0099934
1: -0.0035800, 0.0021775, -0.0035800, 0.0021775, -0.0056476, 0.0056410
2: 0.0069218, 0.0168613, 0.0069218, 0.0168613, -0.0099395, 0.0099395
3: 1.0059226, 1.0071489, 1.0059226, 1.0071489, -0.0012263, 0.0012263
4: -0.0043803, -0.0012098, -0.0043803, -0.0012098, -0.0031706, 0.0031706
5: 0.0035104, 0.0163329, 0.0035104, 0.0163329, -0.0122862, 0.0122755
6: -0.0122887, -0.0025326, -0.0122887, -0.0025326, -0.0097561, 0.0097561
7: -0.0173318, -0.0102117, -0.0173318, -0.0102117, -0.0070415, 0.0070431
8: -0.0150475, -0.0075314, -0.0150475, -0.0075314, -0.0075160, 0.0075160
9: -0.0052566, 0.0032634, -0.0052566, 0.0032634, -0.0085200, 0.0085200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006398, upper bound: 0.0006419
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006398, upper bound: 0.0006419
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0006163, 0.0100718, -0.0006163, 0.0100718, -0.0099899, 0.0100097
1: -0.0035800, 0.0021775, -0.0035800, 0.0021775, -0.0056393, 0.0056490
2: 0.0069218, 0.0168613, 0.0069218, 0.0168613, -0.0099395, 0.0099395
3: 1.0059226, 1.0071489, 1.0059226, 1.0071489, -0.0012263, 0.0012263
4: -0.0043803, -0.0012098, -0.0043803, -0.0012098, -0.0031706, 0.0031706
5: 0.0035104, 0.0163329, 0.0035104, 0.0163329, -0.0122729, 0.0122884
6: -0.0122887, -0.0025326, -0.0122887, -0.0025326, -0.0097561, 0.0097561
7: -0.0173318, -0.0102117, -0.0173318, -0.0102117, -0.0070434, 0.0070411
8: -0.0150475, -0.0075314, -0.0150475, -0.0075314, -0.0075160, 0.0075160
9: -0.0052566, 0.0032634, -0.0052566, 0.0032634, -0.0085200, 0.0085200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 133
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 64
type: RSZ, layer: 3, pos: 162
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 73

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006419, upper bound: 0.0006399
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0006419, upper bound: 0.0006398
time: 0.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.08
Output dim: 3, lower bound: -0.0006398, upper bound: 0.0006419
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.08
Output dim: 3, lower bound: -0.0006398, upper bound: 0.0006419
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.08
Output dim: 3, lower bound: -0.0006419, upper bound: 0.0006399
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.08
Output dim: 3, lower bound: -0.0006419, upper bound: 0.0006398
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.08
Output dim: 3, lower bound: -0.0006398, upper bound: 0.0006419
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.08
Output dim: 3, lower bound: -0.0006398, upper bound: 0.0006419
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.08
Output dim: 3, lower bound: -0.0006419, upper bound: 0.0006399
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.08
Output dim: 3, lower bound: -0.0006419, upper bound: 0.0006398

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.16 + 20.61 = 23.77 seconds
