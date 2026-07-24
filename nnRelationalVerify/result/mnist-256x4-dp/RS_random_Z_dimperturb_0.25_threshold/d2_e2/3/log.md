## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.004636575


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003629, 0.0003629)
1: (-0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0013516, 0.0013516)
2: (0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0019979, 0.0019979)
3: (-0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0014911, 0.0014911)
4: (-0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014732, 0.0014732)
5: (0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0014873, 0.0014873)
6: (0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844)
7: (-0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0031115, 0.0031115)
8: (0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0093538, 0.0093538)
9: (0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0026551, 0.0026551)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.54 = 2.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0063105, upper bound: 0.0063105

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062546, upper bound: 0.0062377
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062377, upper bound: 0.0062546
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 8, lower bound: -0.0062546, upper bound: 0.0062377
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 8, lower bound: -0.0062377, upper bound: 0.0062546

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003531, 0.0003540
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0013187, 0.0013229
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0019541, 0.0019477
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0014577, 0.0014529
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014478, 0.0014434
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0014539, 0.0014491
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0030232, 0.0030336
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0091198, 0.0091495
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0025931, 0.0025844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061535, upper bound: 0.0061620
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061758, upper bound: 0.0061288
time: 0.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003540, 0.0003531
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0013229, 0.0013187
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0019477, 0.0019541
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0014529, 0.0014577
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014434, 0.0014478
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0014491, 0.0014539
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0030336, 0.0030232
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0091495, 0.0091198
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0025844, 0.0025931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061288, upper bound: 0.0061758
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061620, upper bound: 0.0061535
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 8, lower bound: -0.0061535, upper bound: 0.0061620
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 8, lower bound: -0.0061758, upper bound: 0.0061288
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 8, lower bound: -0.0061288, upper bound: 0.0061758
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 8, lower bound: -0.0061620, upper bound: 0.0061535

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003453, 0.0003464
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012728, 0.0012783
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018856, 0.0018774
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0014056, 0.0013994
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014062, 0.0014005
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0014018, 0.0013957
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0029037, 0.0029171
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087938, 0.0088319
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024954, 0.0024842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055306, upper bound: 0.0055235
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055306, upper bound: 0.0055235
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003460, 0.0003462
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012762, 0.0012771
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018838, 0.0018825
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0014042, 0.0014032
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014049, 0.0014041
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0014005, 0.0013995
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0029120, 0.0029141
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0088175, 0.0088234
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024929, 0.0024912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061033, upper bound: 0.0060647
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061112, upper bound: 0.0060352
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003462, 0.0003460
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012771, 0.0012762
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018825, 0.0018838
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0014032, 0.0014042
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014041, 0.0014049
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013995, 0.0014005
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0029141, 0.0029120
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0088234, 0.0088175
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024912, 0.0024929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060093, upper bound: 0.0060085
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059577, upper bound: 0.0060561
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003464, 0.0003453
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012783, 0.0012728
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018774, 0.0018856
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013994, 0.0014056
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014005, 0.0014062
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013957, 0.0014018
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0029171, 0.0029037
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0088319, 0.0087938
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024842, 0.0024954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060701, upper bound: 0.0060908
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060956, upper bound: 0.0060779
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0055306, upper bound: 0.0055235
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0055306, upper bound: 0.0055235
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0061033, upper bound: 0.0060647
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0061112, upper bound: 0.0060352
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0060093, upper bound: 0.0060085
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0059577, upper bound: 0.0060561
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0060701, upper bound: 0.0060908
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 8, lower bound: -0.0060956, upper bound: 0.0060779

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003452, 0.0003464
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012726, 0.0012780
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018852, 0.0018771
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0014053, 0.0013992
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014059, 0.0014003
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0014015, 0.0013955
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0029032, 0.0029164
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087922, 0.0088300
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024949, 0.0024838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050065, upper bound: 0.0051757
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051854, upper bound: 0.0049896
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003453, 0.0003464
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012728, 0.0012781
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018852, 0.0018774
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0014053, 0.0013994
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0014060, 0.0014005
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0014016, 0.0013957
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0029037, 0.0029165
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087938, 0.0088303
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024950, 0.0024842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054608, upper bound: 0.0054461
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054608, upper bound: 0.0054391
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003445, 0.0003449
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012685, 0.0012702
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018734, 0.0018709
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013964, 0.0013945
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013978, 0.0013960
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013927, 0.0013908
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028931, 0.0028972
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087632, 0.0087751
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024787, 0.0024752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059586, upper bound: 0.0057228
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057763, upper bound: 0.0059215
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003447, 0.0003447
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012693, 0.0012693
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018721, 0.0018721
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013955, 0.0013955
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013969, 0.0013969
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013918, 0.0013918
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028952, 0.0028952
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087692, 0.0087692
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024770, 0.0024770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054760, upper bound: 0.0054201
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054760, upper bound: 0.0054201
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003335, 0.0003331
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012197, 0.0012176
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017895, 0.0017927
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013314, 0.0013338
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013528, 0.0013549
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013277, 0.0013301
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027566, 0.0027515
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0084030, 0.0083884
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023551, 0.0023594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053527, upper bound: 0.0053814
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053527, upper bound: 0.0053814
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003333, 0.0003345
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012185, 0.0012241
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017992, 0.0017908
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013387, 0.0013324
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013595, 0.0013536
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013350, 0.0013287
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027536, 0.0027673
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083943, 0.0084338
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023685, 0.0023569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053434, upper bound: 0.0053971
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053434, upper bound: 0.0053971
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003449, 0.0003440
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012701, 0.0012659
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018670, 0.0018734
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013916, 0.0013964
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013934, 0.0013977
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013879, 0.0013927
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028972, 0.0028869
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087750, 0.0087454
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024700, 0.0024787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059506, upper bound: 0.0059308
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059064, upper bound: 0.0059718
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003452, 0.0003438
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012714, 0.0012651
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018658, 0.0018752
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013907, 0.0013978
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013925, 0.0013990
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013870, 0.0013941
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0029002, 0.0028848
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087836, 0.0087395
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024682, 0.0024812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059775, upper bound: 0.0059246
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059234, upper bound: 0.0059560
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0050065, upper bound: 0.0051757
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0051854, upper bound: 0.0049896
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0054608, upper bound: 0.0054461
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0054608, upper bound: 0.0054391
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0059586, upper bound: 0.0057228
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0057763, upper bound: 0.0059215
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0054760, upper bound: 0.0054201
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0054760, upper bound: 0.0054201
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0053527, upper bound: 0.0053814
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0053527, upper bound: 0.0053814
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0053434, upper bound: 0.0053971
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0053434, upper bound: 0.0053971
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0059506, upper bound: 0.0059308
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0059064, upper bound: 0.0059718
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0059775, upper bound: 0.0059246
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 8, lower bound: -0.0059234, upper bound: 0.0059560

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003026, 0.0003112
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010984, 0.0011385
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0016680, 0.0016080
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012373, 0.0011921
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012763, 0.0012347
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012334, 0.0011884
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0024136, 0.0025114
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0075453, 0.0078255
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0021651, 0.0020827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035490, upper bound: 0.0036736
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035490, upper bound: 0.0036736
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003117, 0.0003038
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011411, 0.0011038
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0016161, 0.0016718
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011982, 0.0012402
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012403, 0.0012790
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011945, 0.0012363
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0025177, 0.0024268
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0078435, 0.0075832
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020939, 0.0021704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050244, upper bound: 0.0048166
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050092, upper bound: 0.0048445
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003438, 0.0003451
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012651, 0.0012712
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018749, 0.0018658
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013975, 0.0013907
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013988, 0.0013925
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013938, 0.0013870
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028848, 0.0028997
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087395, 0.0087820
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024808, 0.0024682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052918, upper bound: 0.0052697
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052798, upper bound: 0.0052832
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003440, 0.0003449
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012659, 0.0012699
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018730, 0.0018670
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013961, 0.0013916
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013975, 0.0013934
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013924, 0.0013879
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028869, 0.0028967
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087454, 0.0087734
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024782, 0.0024700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051580, upper bound: 0.0050910
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051042, upper bound: 0.0051570
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003408, 0.0003365
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012550, 0.0012345
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018200, 0.0018507
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013563, 0.0013793
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013607, 0.0013820
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013526, 0.0013757
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028602, 0.0028102
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0086691, 0.0085259
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024055, 0.0024475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052109, upper bound: 0.0051391
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052109, upper bound: 0.0051391
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003361, 0.0003449
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012328, 0.0012702
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018734, 0.0018175
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013964, 0.0013544
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013978, 0.0013590
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013927, 0.0013507
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028061, 0.0028972
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0085140, 0.0087751
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024787, 0.0024020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051790, upper bound: 0.0051771
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051790, upper bound: 0.0051771
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003447, 0.0003447
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012691, 0.0012690
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018717, 0.0018718
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013951, 0.0013952
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013966, 0.0013967
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013914, 0.0013915
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028946, 0.0028945
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087676, 0.0087671
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024764, 0.0024765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0040120, upper bound: 0.0039599
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0040120, upper bound: 0.0039599
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003447, 0.0003447
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012693, 0.0012691
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018718, 0.0018721
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013952, 0.0013955
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013967, 0.0013969
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013915, 0.0013918
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028952, 0.0028946
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0087692, 0.0087676
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024765, 0.0024770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053072, upper bound: 0.0052496
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052948, upper bound: 0.0052618
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003335, 0.0003330
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012195, 0.0012175
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017893, 0.0017923
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013312, 0.0013335
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013526, 0.0013547
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013275, 0.0013298
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027560, 0.0027510
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0084015, 0.0083872
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023547, 0.0023589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038167, upper bound: 0.0038584
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0038167, upper bound: 0.0038584
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003335, 0.0003330
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012197, 0.0012174
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017892, 0.0017927
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013312, 0.0013338
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013526, 0.0013549
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013275, 0.0013301
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027566, 0.0027509
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0084030, 0.0083869
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023547, 0.0023594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052618, upper bound: 0.0052948
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052662, upper bound: 0.0052948
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003332, 0.0003344
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012183, 0.0012239
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017990, 0.0017905
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013385, 0.0013321
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013593, 0.0013534
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013348, 0.0013284
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027530, 0.0027668
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083929, 0.0084325
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023680, 0.0023564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050471, upper bound: 0.0050625
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050024, upper bound: 0.0050956
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003333, 0.0003344
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012185, 0.0012239
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017989, 0.0017908
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013385, 0.0013324
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013593, 0.0013536
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013348, 0.0013287
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027536, 0.0027668
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083943, 0.0084323
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023680, 0.0023569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050471, upper bound: 0.0050625
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050024, upper bound: 0.0050956
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003328, 0.0003310
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012157, 0.0012073
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017741, 0.0017867
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013198, 0.0013293
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013420, 0.0013508
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013161, 0.0013256
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027469, 0.0027263
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083751, 0.0083163
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023339, 0.0023512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054349, upper bound: 0.0053767
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054349, upper bound: 0.0053767
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003319, 0.0003324
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012115, 0.0012137
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017836, 0.0017804
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013270, 0.0013246
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013486, 0.0013464
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013233, 0.0013209
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027366, 0.0027418
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083458, 0.0083607
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023470, 0.0023426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053792, upper bound: 0.0056247
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055591, upper bound: 0.0054230
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003331, 0.0003309
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012172, 0.0012065
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017728, 0.0017889
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013189, 0.0013310
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013412, 0.0013523
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013152, 0.0013273
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027505, 0.0027243
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083856, 0.0083103
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023322, 0.0023543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052832, upper bound: 0.0052798
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052832, upper bound: 0.0052798
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003322, 0.0003321
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012128, 0.0012125
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017819, 0.0017822
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013257, 0.0013260
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013474, 0.0013477
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013220, 0.0013223
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027396, 0.0027390
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083544, 0.0083526
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023446, 0.0023451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0053950, upper bound: 0.0056144
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0055709, upper bound: 0.0054032
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0035490, upper bound: 0.0036736
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0035490, upper bound: 0.0036736
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0050244, upper bound: 0.0048166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0050092, upper bound: 0.0048445
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052918, upper bound: 0.0052697
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052798, upper bound: 0.0052832
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0051580, upper bound: 0.0050910
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0051042, upper bound: 0.0051570
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052109, upper bound: 0.0051391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052109, upper bound: 0.0051391
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0051790, upper bound: 0.0051771
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0051790, upper bound: 0.0051771
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0040120, upper bound: 0.0039599
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0040120, upper bound: 0.0039599
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0053072, upper bound: 0.0052496
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052948, upper bound: 0.0052618
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0038167, upper bound: 0.0038584
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0038167, upper bound: 0.0038584
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052618, upper bound: 0.0052948
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052662, upper bound: 0.0052948
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0050471, upper bound: 0.0050625
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0050024, upper bound: 0.0050956
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0050471, upper bound: 0.0050625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0050024, upper bound: 0.0050956
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0054349, upper bound: 0.0053767
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0054349, upper bound: 0.0053767
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0053792, upper bound: 0.0056247
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0055591, upper bound: 0.0054230
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052832, upper bound: 0.0052798
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0052832, upper bound: 0.0052798
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0053950, upper bound: 0.0056144
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 8, lower bound: -0.0055709, upper bound: 0.0054032

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002984, 0.0002895
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010865, 0.0010449
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015246, 0.0015869
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011282, 0.0011750
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011865, 0.0012297
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011244, 0.0011712
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023509, 0.0022493
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0074501, 0.0071592
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019528, 0.0020383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034440, upper bound: 0.0033541
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034440, upper bound: 0.0033541
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002974, 0.0002904
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010821, 0.0010494
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015313, 0.0015804
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011332, 0.0011701
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011911, 0.0012251
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011294, 0.0011662
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023402, 0.0022602
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0074195, 0.0071903
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019620, 0.0020293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049231, upper bound: 0.0047593
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049231, upper bound: 0.0047533
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003321, 0.0003321
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012125, 0.0012126
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017819, 0.0017819
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013257, 0.0013257
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013475, 0.0013474
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013220, 0.0013220
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027390, 0.0027391
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083526, 0.0083529
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023447, 0.0023446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049923, upper bound: 0.0049089
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049169, upper bound: 0.0049799
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003309, 0.0003331
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012065, 0.0012170
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017886, 0.0017728
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013307, 0.0013189
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013521, 0.0013412
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013270, 0.0013152
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027243, 0.0027499
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083103, 0.0083841
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023538, 0.0023322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049849, upper bound: 0.0049363
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049091, upper bound: 0.0049974
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003406, 0.0003364
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012539, 0.0012343
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018197, 0.0018490
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013560, 0.0013781
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013605, 0.0013809
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013524, 0.0013744
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028575, 0.0028097
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0086614, 0.0085241
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024050, 0.0024453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031910, upper bound: 0.0032159
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0031910, upper bound: 0.0032159
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003355, 0.0003449
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012303, 0.0012699
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018730, 0.0018137
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013961, 0.0013515
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013975, 0.0013563
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013924, 0.0013479
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027999, 0.0028967
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0084962, 0.0087734
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024782, 0.0023967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045719, upper bound: 0.0048058
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047464, upper bound: 0.0046310
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003406, 0.0003357
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012544, 0.0012312
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018150, 0.0018498
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013525, 0.0013786
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013573, 0.0013814
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013489, 0.0013750
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028587, 0.0028021
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0086648, 0.0085025
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023986, 0.0024463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032161, upper bound: 0.0031907
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032161, upper bound: 0.0031907
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003401, 0.0003365
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012517, 0.0012345
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018200, 0.0018457
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013563, 0.0013756
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013607, 0.0013785
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013526, 0.0013719
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028521, 0.0028102
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0086457, 0.0085259
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024055, 0.0024407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032161, upper bound: 0.0031907
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032161, upper bound: 0.0031907
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003359, 0.0003441
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012323, 0.0012670
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018686, 0.0018167
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013928, 0.0013538
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013945, 0.0013584
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013891, 0.0013502
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0028049, 0.0028895
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0085105, 0.0087529
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024722, 0.0024009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0045970, upper bound: 0.0048331
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048651, upper bound: 0.0046339
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003353, 0.0003449
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012295, 0.0012702
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0018734, 0.0018125
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013964, 0.0013506
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013978, 0.0013555
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013927, 0.0013470
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027979, 0.0028972
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0084906, 0.0087751
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0024787, 0.0023951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050336, upper bound: 0.0050011
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049973, upper bound: 0.0050385
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003331, 0.0003317
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012172, 0.0012105
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017788, 0.0017889
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013234, 0.0013309
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013454, 0.0013523
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013197, 0.0013272
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027504, 0.0027340
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083854, 0.0083385
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023404, 0.0023542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047810, upper bound: 0.0048905
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049629, upper bound: 0.0047325
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003318, 0.0003319
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012107, 0.0012116
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017804, 0.0017792
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013246, 0.0013236
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013465, 0.0013456
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013209, 0.0013199
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027346, 0.0027366
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083400, 0.0083460
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023426, 0.0023409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047601, upper bound: 0.0049066
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049351, upper bound: 0.0047505
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003320, 0.0003317
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012118, 0.0012105
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017788, 0.0017807
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013234, 0.0013248
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013454, 0.0013467
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013197, 0.0013211
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027372, 0.0027340
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083475, 0.0083385
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023404, 0.0023431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049702, upper bound: 0.0049284
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048838, upper bound: 0.0049996
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003322, 0.0003315
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012128, 0.0012096
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017776, 0.0017823
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013224, 0.0013260
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013445, 0.0013477
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013187, 0.0013223
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027397, 0.0027320
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083546, 0.0083326
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023387, 0.0023452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037613, upper bound: 0.0037993
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0037613, upper bound: 0.0037993
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003289, 0.0003261
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011998, 0.0011865
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017429, 0.0017628
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012963, 0.0013113
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013204, 0.0013342
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012927, 0.0013076
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027079, 0.0026754
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082636, 0.0081707
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022911, 0.0023184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030435, upper bound: 0.0030793
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030435, upper bound: 0.0030793
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003249, 0.0003344
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011808, 0.0012239
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017990, 0.0017344
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013385, 0.0012900
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013593, 0.0013145
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013348, 0.0012863
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026616, 0.0027668
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0081310, 0.0084325
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023680, 0.0022794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030435, upper bound: 0.0030793
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030435, upper bound: 0.0030793
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003290, 0.0003261
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012000, 0.0011865
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017428, 0.0017631
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012963, 0.0013116
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013204, 0.0013344
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012927, 0.0013079
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027085, 0.0026754
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082651, 0.0081705
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022910, 0.0023189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049651, upper bound: 0.0049543
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049698, upper bound: 0.0049488
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003249, 0.0003344
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011810, 0.0012239
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017989, 0.0017347
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013385, 0.0012902
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013593, 0.0013147
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013348, 0.0012866
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026622, 0.0027668
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0081325, 0.0084323
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023680, 0.0022799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044604, upper bound: 0.0047282
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046382, upper bound: 0.0045686
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003326, 0.0003303
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012151, 0.0012039
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017690, 0.0017857
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013160, 0.0013285
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013385, 0.0013501
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013123, 0.0013248
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027452, 0.0027180
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083705, 0.0082925
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023270, 0.0023499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050657, upper bound: 0.0049746
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0050192, upper bound: 0.0050157
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003321, 0.0003310
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012123, 0.0012073
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017741, 0.0017816
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013198, 0.0013255
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013420, 0.0013472
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013161, 0.0013218
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027386, 0.0027263
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083514, 0.0083163
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023339, 0.0023442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048551, upper bound: 0.0050454
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0051046, upper bound: 0.0048368
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002881, 0.0002972
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010388, 0.0010816
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015796, 0.0015155
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011695, 0.0011213
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012245, 0.0011801
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011656, 0.0011175
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022346, 0.0023390
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071166, 0.0074158
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020283, 0.0019403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048228, upper bound: 0.0050901
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048228, upper bound: 0.0050901
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002955, 0.0002885
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010737, 0.0010409
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015187, 0.0015677
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011237, 0.0011606
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011823, 0.0012163
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011199, 0.0011567
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023196, 0.0022397
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073603, 0.0071314
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019447, 0.0020120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0054105, upper bound: 0.0050927
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0052382, upper bound: 0.0052808
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003331, 0.0003308
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012170, 0.0012062
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017724, 0.0017886
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013185, 0.0013307
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013409, 0.0013521
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013148, 0.0013270
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027499, 0.0027235
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083841, 0.0083084
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023316, 0.0023538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049974, upper bound: 0.0049091
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049363, upper bound: 0.0049849
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003331, 0.0003308
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0012172, 0.0012062
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017725, 0.0017889
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013186, 0.0013310
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013410, 0.0013523
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013149, 0.0013273
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027505, 0.0027237
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0083856, 0.0083088
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023317, 0.0023543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049974, upper bound: 0.0049091
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049363, upper bound: 0.0049849
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002884, 0.0002969
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010400, 0.0010802
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015775, 0.0015173
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011680, 0.0011227
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012231, 0.0011814
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011641, 0.0011189
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022376, 0.0023356
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071252, 0.0074061
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020254, 0.0019429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048300, upper bound: 0.0050808
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0048300, upper bound: 0.0050808
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002957, 0.0002883
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010746, 0.0010398
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015169, 0.0015690
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011224, 0.0011616
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011811, 0.0012172
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011186, 0.0011577
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023218, 0.0022369
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073666, 0.0071234
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019423, 0.0020138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049244, upper bound: 0.0047802
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0049244, upper bound: 0.0047802
time: 0.55 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0034440, upper bound: 0.0033541
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0034440, upper bound: 0.0033541
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049231, upper bound: 0.0047593
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049231, upper bound: 0.0047533
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049923, upper bound: 0.0049089
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049169, upper bound: 0.0049799
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049849, upper bound: 0.0049363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049091, upper bound: 0.0049974
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0031910, upper bound: 0.0032159
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0031910, upper bound: 0.0032159
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0045719, upper bound: 0.0048058
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0047464, upper bound: 0.0046310
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0032161, upper bound: 0.0031907
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0032161, upper bound: 0.0031907
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0032161, upper bound: 0.0031907
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0032161, upper bound: 0.0031907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0045970, upper bound: 0.0048331
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0048651, upper bound: 0.0046339
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0050336, upper bound: 0.0050011
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049973, upper bound: 0.0050385
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0047810, upper bound: 0.0048905
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049629, upper bound: 0.0047325
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0047601, upper bound: 0.0049066
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049351, upper bound: 0.0047505
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049702, upper bound: 0.0049284
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0048838, upper bound: 0.0049996
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0037613, upper bound: 0.0037993
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0037613, upper bound: 0.0037993
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0030435, upper bound: 0.0030793
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0030435, upper bound: 0.0030793
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0030435, upper bound: 0.0030793
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0030435, upper bound: 0.0030793
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049651, upper bound: 0.0049543
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049698, upper bound: 0.0049488
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0044604, upper bound: 0.0047282
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0046382, upper bound: 0.0045686
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0050657, upper bound: 0.0049746
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0050192, upper bound: 0.0050157
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0048551, upper bound: 0.0050454
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0051046, upper bound: 0.0048368
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0048228, upper bound: 0.0050901
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0048228, upper bound: 0.0050901
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0054105, upper bound: 0.0050927
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0052382, upper bound: 0.0052808
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049974, upper bound: 0.0049091
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049363, upper bound: 0.0049849
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049974, upper bound: 0.0049091
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049363, upper bound: 0.0049849
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0048300, upper bound: 0.0050808
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0048300, upper bound: 0.0050808
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049244, upper bound: 0.0047802
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 8, lower bound: -0.0049244, upper bound: 0.0047802

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002959, 0.0002893
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010754, 0.0010443
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015237, 0.0015704
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011275, 0.0011626
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011858, 0.0012182
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011237, 0.0011587
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023239, 0.0022478
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073727, 0.0071547
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019515, 0.0020156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033816, upper bound: 0.0033128
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033816, upper bound: 0.0033128
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002963, 0.0002889
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010770, 0.0010428
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015214, 0.0015728
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011258, 0.0011644
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011843, 0.0012199
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011220, 0.0011605
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023278, 0.0022442
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073839, 0.0071444
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019485, 0.0020189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0046194, upper bound: 0.0043962
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045662, upper bound: 0.0044766
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003282, 0.0003238
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011961, 0.0011758
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017269, 0.0017573
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012843, 0.0013072
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013093, 0.0013304
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012807, 0.0013035
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026990, 0.0026494
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082379, 0.0080961
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022692, 0.0023109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044880, upper bound: 0.0045607
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0046306, upper bound: 0.0043666
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003238, 0.0003321
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011758, 0.0012126
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017819, 0.0017268
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013257, 0.0012843
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013475, 0.0013093
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013220, 0.0012807
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026494, 0.0027391
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0080958, 0.0083529
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023447, 0.0022691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0043997, upper bound: 0.0046250
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045669, upper bound: 0.0044364
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003278, 0.0003248
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011944, 0.0011803
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017336, 0.0017547
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012894, 0.0013052
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013140, 0.0013286
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012857, 0.0013016
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026947, 0.0026603
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082257, 0.0081273
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022783, 0.0023073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029812, upper bound: 0.0030229
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029812, upper bound: 0.0030229
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003225, 0.0003331
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011697, 0.0012170
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017886, 0.0017178
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013307, 0.0012775
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013521, 0.0013030
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013270, 0.0012739
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026346, 0.0027499
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0080535, 0.0083841
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023538, 0.0022567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043737, upper bound: 0.0046444
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045620, upper bound: 0.0044792
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002931, 0.0003097
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010593, 0.0011323
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0016588, 0.0015494
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012304, 0.0011481
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012699, 0.0011941
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012265, 0.0011444
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023183, 0.0024964
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0072721, 0.0077826
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0021525, 0.0020024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044125, upper bound: 0.0046249
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0043789, upper bound: 0.0046425
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003022, 0.0003022
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011020, 0.0010975
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0016066, 0.0016133
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011911, 0.0011962
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012337, 0.0012384
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011873, 0.0011924
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0024223, 0.0024113
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0075702, 0.0075389
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020808, 0.0020901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0028157, upper bound: 0.0028013
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0028157, upper bound: 0.0028013
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002934, 0.0003076
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010609, 0.0011231
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0016450, 0.0015518
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012200, 0.0011499
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012603, 0.0011957
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012162, 0.0011462
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023222, 0.0024740
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0072833, 0.0077182
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0021336, 0.0020057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0028031, upper bound: 0.0028153
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0028031, upper bound: 0.0028153
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003019, 0.0003014
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011008, 0.0010941
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0016016, 0.0016115
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011873, 0.0011948
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012302, 0.0012371
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011836, 0.0011911
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0024194, 0.0024032
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0075619, 0.0075155
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020740, 0.0020876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047175, upper bound: 0.0044690
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046786, upper bound: 0.0044975
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003237, 0.0003319
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011755, 0.0012116
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017804, 0.0017264
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013246, 0.0012840
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013464, 0.0013090
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013209, 0.0012804
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026487, 0.0027367
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0080939, 0.0083459
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023427, 0.0022686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044480, upper bound: 0.0046494
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0047175, upper bound: 0.0044690
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003225, 0.0003322
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011697, 0.0012128
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017823, 0.0017177
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013260, 0.0012775
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013477, 0.0013030
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013223, 0.0012738
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026345, 0.0027397
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0080533, 0.0083546
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023452, 0.0022566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030116, upper bound: 0.0029943
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030116, upper bound: 0.0029943
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002893, 0.0002939
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010445, 0.0010662
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015566, 0.0015240
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011522, 0.0011277
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012086, 0.0011860
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011484, 0.0011239
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022484, 0.0023014
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071561, 0.0073084
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019967, 0.0019520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044950, upper bound: 0.0045167
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044244, upper bound: 0.0046066
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002982, 0.0002879
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010860, 0.0010378
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015140, 0.0015862
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011202, 0.0011745
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011791, 0.0012291
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011164, 0.0011706
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023497, 0.0022320
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0074466, 0.0071095
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019382, 0.0020373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046543, upper bound: 0.0043605
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045984, upper bound: 0.0044361
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002879, 0.0002964
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010380, 0.0010777
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015738, 0.0015142
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011652, 0.0011204
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012206, 0.0011792
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011613, 0.0011166
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022325, 0.0023295
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071108, 0.0073887
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020203, 0.0019386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033008, upper bound: 0.0033645
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033008, upper bound: 0.0033645
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002968, 0.0002881
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010794, 0.0010389
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015156, 0.0015762
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011214, 0.0011670
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011802, 0.0012222
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011176, 0.0011631
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023335, 0.0022346
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0074002, 0.0071169
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019404, 0.0020237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034038, upper bound: 0.0032830
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034038, upper bound: 0.0032830
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003279, 0.0003234
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011947, 0.0011738
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017238, 0.0017551
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012820, 0.0013056
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013072, 0.0013289
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012784, 0.0013019
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026955, 0.0026444
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082279, 0.0080817
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022649, 0.0023080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044682, upper bound: 0.0045820
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0046104, upper bound: 0.0043846
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003237, 0.0003317
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011750, 0.0012105
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017788, 0.0017257
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013234, 0.0012835
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013454, 0.0013085
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013197, 0.0012798
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026476, 0.0027340
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0080906, 0.0083385
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023404, 0.0022676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029862, upper bound: 0.0030140
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029862, upper bound: 0.0030140
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003275, 0.0003248
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011930, 0.0011803
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017335, 0.0017527
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012893, 0.0013037
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013139, 0.0013272
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012857, 0.0013000
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026914, 0.0026602
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082163, 0.0081271
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022783, 0.0023045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029800, upper bound: 0.0030235
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0029800, upper bound: 0.0030235
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003277, 0.0003244
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011938, 0.0011787
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017313, 0.0017538
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012876, 0.0013046
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013124, 0.0013280
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012840, 0.0013009
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026933, 0.0026565
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082217, 0.0081164
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022751, 0.0023061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044364, upper bound: 0.0045975
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0046102, upper bound: 0.0044188
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002810, 0.0002993
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010115, 0.0010909
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015935, 0.0014746
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011800, 0.0010906
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012343, 0.0011518
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011761, 0.0010869
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0021680, 0.0023616
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0069259, 0.0074809
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020474, 0.0018843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026278, upper bound: 0.0027054
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026278, upper bound: 0.0027054
time: 0.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002872, 0.0002904
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010405, 0.0010494
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015313, 0.0015181
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011332, 0.0011233
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011911, 0.0011819
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011294, 0.0011195
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022388, 0.0022603
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071287, 0.0071905
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019620, 0.0019439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045167, upper bound: 0.0044950
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0045308, upper bound: 0.0044947
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003277, 0.0003219
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011943, 0.0011671
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017139, 0.0017546
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012746, 0.0013052
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013003, 0.0013285
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012710, 0.0013015
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026946, 0.0026283
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082255, 0.0080355
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022514, 0.0023072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030196, upper bound: 0.0029887
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030196, upper bound: 0.0029887
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003243, 0.0003303
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011783, 0.0012039
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017690, 0.0017306
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013160, 0.0012871
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013385, 0.0013119
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013123, 0.0012835
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026555, 0.0027180
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0081134, 0.0082925
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023270, 0.0022743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030196, upper bound: 0.0029887
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030196, upper bound: 0.0029887
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002882, 0.0002963
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010392, 0.0010772
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015730, 0.0015160
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011646, 0.0011217
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012200, 0.0011805
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011607, 0.0011179
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022354, 0.0023283
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071190, 0.0073852
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020193, 0.0019411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044945, upper bound: 0.0046437
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044480, upper bound: 0.0046769
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002972, 0.0002872
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010813, 0.0010346
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015091, 0.0015791
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011166, 0.0011691
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011757, 0.0012242
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011128, 0.0011653
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023382, 0.0022242
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0074135, 0.0070870
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019316, 0.0020276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034221, upper bound: 0.0032916
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0034221, upper bound: 0.0032916
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002879, 0.0002964
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010378, 0.0010778
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015738, 0.0015139
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011652, 0.0011201
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012206, 0.0011790
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011613, 0.0011164
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022320, 0.0023296
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071093, 0.0073889
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020204, 0.0019382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044653, upper bound: 0.0046664
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0044234, upper bound: 0.0047070
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002873, 0.0002972
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010350, 0.0010816
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015796, 0.0015097
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011695, 0.0011170
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012245, 0.0011761
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011656, 0.0011132
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022252, 0.0023390
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0070897, 0.0074158
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020283, 0.0019325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032945, upper bound: 0.0033851
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032945, upper bound: 0.0033851
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002915, 0.0002803
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010620, 0.0010093
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0014713, 0.0015502
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0010881, 0.0011474
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011494, 0.0012042
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0010843, 0.0011436
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022911, 0.0021625
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0072786, 0.0069102
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0018797, 0.0019880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0046249, upper bound: 0.0044125
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0046249, upper bound: 0.0044125
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002873, 0.0002885
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010420, 0.0010409
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015187, 0.0015203
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011237, 0.0011249
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011823, 0.0011834
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011199, 0.0011211
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022424, 0.0022397
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071391, 0.0071314
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019447, 0.0019469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046603, upper bound: 0.0045048
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046603, upper bound: 0.0045048
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003282, 0.0003225
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011965, 0.0011695
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017174, 0.0017578
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012772, 0.0013076
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013027, 0.0013308
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012736, 0.0013039
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026998, 0.0026339
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082404, 0.0080516
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022561, 0.0023116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044792, upper bound: 0.0045620
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046444, upper bound: 0.0043737
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003248, 0.0003308
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011803, 0.0012062
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017724, 0.0017336
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013185, 0.0012894
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013409, 0.0013140
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013148, 0.0012857
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026603, 0.0027235
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0081272, 0.0083084
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023316, 0.0022783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030229, upper bound: 0.0029812
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030229, upper bound: 0.0029812
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003283, 0.0003225
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011967, 0.0011695
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017175, 0.0017581
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0012772, 0.0013078
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013028, 0.0013310
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0012736, 0.0013042
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0027004, 0.0026340
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0082419, 0.0080520
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0022562, 0.0023121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0044792, upper bound: 0.0045620
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0046444, upper bound: 0.0043737
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0003248, 0.0003308
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0011805, 0.0012062
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0017725, 0.0017339
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0013186, 0.0012896
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0013410, 0.0013142
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0013149, 0.0012860
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0026609, 0.0027237
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0081287, 0.0083088
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0023317, 0.0022788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030229, upper bound: 0.0029812
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0030229, upper bound: 0.0029812
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002881, 0.0002961
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010390, 0.0010764
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015718, 0.0015157
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011636, 0.0011215
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012191, 0.0011803
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011598, 0.0011177
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022350, 0.0023263
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071178, 0.0073793
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020175, 0.0019407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032936, upper bound: 0.0033851
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032936, upper bound: 0.0033851
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002875, 0.0002969
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010362, 0.0010802
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015775, 0.0015116
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011680, 0.0011184
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012231, 0.0011774
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011641, 0.0011146
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022282, 0.0023356
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0070984, 0.0074061
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020254, 0.0019350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032936, upper bound: 0.0033851
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0032936, upper bound: 0.0033851
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002957, 0.0002883
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010744, 0.0010396
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015166, 0.0015688
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011222, 0.0011614
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011809, 0.0012171
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011184, 0.0011575
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023213, 0.0022363
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073654, 0.0071219
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019419, 0.0020134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033810, upper bound: 0.0032938
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033810, upper bound: 0.0032938
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002957, 0.0002883
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010746, 0.0010396
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015167, 0.0015690
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011222, 0.0011616
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011810, 0.0012172
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011184, 0.0011577
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023218, 0.0022364
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073666, 0.0071221
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019419, 0.0020138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033810, upper bound: 0.0032938
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033810, upper bound: 0.0032938
time: 0.59 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0033816, upper bound: 0.0033128
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0033816, upper bound: 0.0033128
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046194, upper bound: 0.0043962
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0045662, upper bound: 0.0044766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044880, upper bound: 0.0045607
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046306, upper bound: 0.0043666
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0043997, upper bound: 0.0046250
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0045669, upper bound: 0.0044364
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0029812, upper bound: 0.0030229
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0029812, upper bound: 0.0030229
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0043737, upper bound: 0.0046444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0045620, upper bound: 0.0044792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044125, upper bound: 0.0046249
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0043789, upper bound: 0.0046425
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0028157, upper bound: 0.0028013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0028157, upper bound: 0.0028013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0028031, upper bound: 0.0028153
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0028031, upper bound: 0.0028153
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0047175, upper bound: 0.0044690
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046786, upper bound: 0.0044975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044480, upper bound: 0.0046494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0047175, upper bound: 0.0044690
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030116, upper bound: 0.0029943
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030116, upper bound: 0.0029943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044950, upper bound: 0.0045167
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044244, upper bound: 0.0046066
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046543, upper bound: 0.0043605
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0045984, upper bound: 0.0044361
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0033008, upper bound: 0.0033645
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0033008, upper bound: 0.0033645
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0034038, upper bound: 0.0032830
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0034038, upper bound: 0.0032830
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044682, upper bound: 0.0045820
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046104, upper bound: 0.0043846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0029862, upper bound: 0.0030140
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0029862, upper bound: 0.0030140
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0029800, upper bound: 0.0030235
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0029800, upper bound: 0.0030235
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044364, upper bound: 0.0045975
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046102, upper bound: 0.0044188
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0026278, upper bound: 0.0027054
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0026278, upper bound: 0.0027054
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0045167, upper bound: 0.0044950
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0045308, upper bound: 0.0044947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030196, upper bound: 0.0029887
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030196, upper bound: 0.0029887
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030196, upper bound: 0.0029887
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030196, upper bound: 0.0029887
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044945, upper bound: 0.0046437
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044480, upper bound: 0.0046769
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0034221, upper bound: 0.0032916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0034221, upper bound: 0.0032916
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044653, upper bound: 0.0046664
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044234, upper bound: 0.0047070
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0032945, upper bound: 0.0033851
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0032945, upper bound: 0.0033851
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046249, upper bound: 0.0044125
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046249, upper bound: 0.0044125
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046603, upper bound: 0.0045048
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046603, upper bound: 0.0045048
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044792, upper bound: 0.0045620
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046444, upper bound: 0.0043737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030229, upper bound: 0.0029812
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030229, upper bound: 0.0029812
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0044792, upper bound: 0.0045620
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0046444, upper bound: 0.0043737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030229, upper bound: 0.0029812
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0030229, upper bound: 0.0029812
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0032936, upper bound: 0.0033851
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0032936, upper bound: 0.0033851
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0032936, upper bound: 0.0033851
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0032936, upper bound: 0.0033851
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0033810, upper bound: 0.0032938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0033810, upper bound: 0.0032938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0033810, upper bound: 0.0032938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 8, lower bound: -0.0033810, upper bound: 0.0032938

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002787, 0.0002982
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010021, 0.0010862
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015865, 0.0014605
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011747, 0.0010800
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012294, 0.0011420
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011708, 0.0010763
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0021450, 0.0023501
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0068599, 0.0074479
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020377, 0.0018649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025737, upper bound: 0.0026486
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025737, upper bound: 0.0026486
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002789, 0.0002979
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010029, 0.0010849
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015846, 0.0014618
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011733, 0.0010809
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012280, 0.0011428
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011694, 0.0010772
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0021470, 0.0023470
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0068658, 0.0074390
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020351, 0.0018666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025743, upper bound: 0.0026410
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025743, upper bound: 0.0026410
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002892, 0.0002873
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010509, 0.0010350
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015098, 0.0015336
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011170, 0.0011349
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011761, 0.0011926
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011132, 0.0011311
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022640, 0.0022252
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0072011, 0.0070899
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019325, 0.0019652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026424, upper bound: 0.0025725
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026424, upper bound: 0.0025725
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002877, 0.0002875
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010442, 0.0010362
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015116, 0.0015236
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011184, 0.0011274
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011774, 0.0011857
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011146, 0.0011236
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022477, 0.0022283
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071543, 0.0070986
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019350, 0.0019514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026357, upper bound: 0.0025811
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026357, upper bound: 0.0025811
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002799, 0.0002943
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010073, 0.0010678
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015589, 0.0014683
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011540, 0.0010859
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012102, 0.0011474
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011502, 0.0010821
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0021577, 0.0023054
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0068965, 0.0073195
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020000, 0.0018756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026061, upper bound: 0.0026158
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026061, upper bound: 0.0026158
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002887, 0.0002881
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010489, 0.0010388
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015155, 0.0015306
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011213, 0.0011327
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011801, 0.0011906
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011175, 0.0011289
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022592, 0.0022346
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071873, 0.0071167
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019404, 0.0019611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026424, upper bound: 0.0025725
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026424, upper bound: 0.0025725
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002944, 0.0002796
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010754, 0.0010062
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0014666, 0.0015703
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0010846, 0.0011626
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011462, 0.0012181
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0010808, 0.0011587
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023239, 0.0021548
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073726, 0.0068883
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0018732, 0.0020156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026488, upper bound: 0.0025718
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026488, upper bound: 0.0025718
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002834, 0.0002880
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010240, 0.0010456
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015256, 0.0014934
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011290, 0.0011047
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011871, 0.0011648
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011252, 0.0011009
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0021985, 0.0022511
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0070133, 0.0071640
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019543, 0.0019100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026048, upper bound: 0.0026156
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026048, upper bound: 0.0026156
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002799, 0.0002963
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010074, 0.0010772
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015730, 0.0014684
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011646, 0.0010859
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012200, 0.0011474
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011607, 0.0010822
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0021579, 0.0023283
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0068968, 0.0073852
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020193, 0.0018758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026048, upper bound: 0.0026156
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026048, upper bound: 0.0026156
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002835, 0.0002881
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010245, 0.0010460
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015262, 0.0014941
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011294, 0.0011052
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011876, 0.0011653
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011256, 0.0011015
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0021997, 0.0022521
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0070167, 0.0071668
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019551, 0.0019110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025822, upper bound: 0.0026191
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025822, upper bound: 0.0026191
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002796, 0.0002964
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010060, 0.0010778
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015738, 0.0014663
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011652, 0.0010844
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0012206, 0.0011460
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011613, 0.0010806
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0021545, 0.0023296
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0068871, 0.0073889
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0020204, 0.0018729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025822, upper bound: 0.0026191
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0025822, upper bound: 0.0026191
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002866, 0.0002877
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010391, 0.0010371
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015129, 0.0015159
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011194, 0.0011216
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011783, 0.0011804
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011156, 0.0011178
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022352, 0.0022304
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071184, 0.0071046
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019368, 0.0019409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026354, upper bound: 0.0025829
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026354, upper bound: 0.0025829
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002864, 0.0002885
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010380, 0.0010409
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0015187, 0.0015143
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0011237, 0.0011205
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011823, 0.0011793
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0011199, 0.0011167
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0022327, 0.0022397
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0071113, 0.0071314
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0019447, 0.0019388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026354, upper bound: 0.0025829
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026354, upper bound: 0.0025829
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002936, 0.0002787
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010718, 0.0010018
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0014601, 0.0015648
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0010797, 0.0011584
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011417, 0.0012144
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0010760, 0.0011546
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023149, 0.0021443
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073470, 0.0068582
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0018643, 0.0020080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026486, upper bound: 0.0025737
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026486, upper bound: 0.0025737
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0002934, 0.0001556, -0.0002934, 0.0001556, -0.0002937, 0.0002787
1: -0.0000533, 0.0015601, -0.0000533, 0.0015601, -0.0010719, 0.0010019
2: 0.0140036, 0.0164198, 0.0140036, 0.0164198, -0.0014602, 0.0015651
3: -0.0000968, 0.0017201, -0.0000968, 0.0017201, -0.0010798, 0.0011586
4: -0.0044689, -0.0027930, -0.0044689, -0.0027930, -0.0011418, 0.0012145
5: 0.0078415, 0.0096552, 0.0078415, 0.0096552, -0.0010761, 0.0011548
6: 0.0092893, 0.0099737, 0.0092893, 0.0099737, -0.0006844, 0.0006844
7: -0.0193597, -0.0154226, -0.0193597, -0.0154226, -0.0023154, 0.0021444
8: 0.9683230, 0.9796035, 0.9683230, 0.9796035, -0.0073483, 0.0068586
9: 0.0036477, 0.0069630, 0.0036477, 0.0069630, -0.0018645, 0.0020084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026486, upper bound: 0.0025737
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0026486, upper bound: 0.0025737
time: 0.49 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0025737, upper bound: 0.0026486
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0025737, upper bound: 0.0026486
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0025743, upper bound: 0.0026410
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0025743, upper bound: 0.0026410
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026424, upper bound: 0.0025725
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026424, upper bound: 0.0025725
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026357, upper bound: 0.0025811
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026357, upper bound: 0.0025811
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026061, upper bound: 0.0026158
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026061, upper bound: 0.0026158
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026424, upper bound: 0.0025725
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026424, upper bound: 0.0025725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026488, upper bound: 0.0025718
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026488, upper bound: 0.0025718
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026048, upper bound: 0.0026156
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026048, upper bound: 0.0026156
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026048, upper bound: 0.0026156
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026048, upper bound: 0.0026156
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0025822, upper bound: 0.0026191
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0025822, upper bound: 0.0026191
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0025822, upper bound: 0.0026191
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0025822, upper bound: 0.0026191
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026354, upper bound: 0.0025829
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026354, upper bound: 0.0025829
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026354, upper bound: 0.0025829
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026354, upper bound: 0.0025829
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026486, upper bound: 0.0025737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026486, upper bound: 0.0025737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026486, upper bound: 0.0025737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.20
Output dim: 8, lower bound: -0.0026486, upper bound: 0.0025737

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.77 + 314.46 = 317.22 seconds
