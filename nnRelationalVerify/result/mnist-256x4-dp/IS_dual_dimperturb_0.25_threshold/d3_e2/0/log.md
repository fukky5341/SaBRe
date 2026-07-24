## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00079287


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0017674, -0.0000191, -0.0017674, -0.0000191, -0.0013817, 0.0013817)
1: (-0.0043648, -0.0037832, -0.0043648, -0.0037832, -0.0004907, 0.0004907)
2: (0.0122873, 0.0146482, 0.0122873, 0.0146482, -0.0018064, 0.0018064)
3: (1.0079025, 1.0093747, 1.0079025, 1.0093747, -0.0014722, 0.0014722)
4: (-0.0039948, -0.0036048, -0.0039948, -0.0036048, -0.0002889, 0.0002889)
5: (0.0025945, 0.0039430, 0.0025945, 0.0039430, -0.0010606, 0.0010606)
6: (-0.0024891, -0.0023534, -0.0024891, -0.0023534, -0.0001357, 0.0001357)
7: (-0.0129936, -0.0107985, -0.0129936, -0.0107985, -0.0021497, 0.0021497)
8: (-0.0107108, -0.0064623, -0.0107108, -0.0064623, -0.0030888, 0.0030888)
9: (-0.0010973, 0.0010199, -0.0010973, 0.0010199, -0.0015101, 0.0015101)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.51 + 1.52 = 3.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0008316, upper bound: 0.0008316

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008073, upper bound: 0.0008108
time: 0.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008194, upper bound: 0.0008194
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.46 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -0.0008073, upper bound: 0.0008108
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -0.0008194, upper bound: 0.0008194

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0018158, -0.0000739, -0.0017673, -0.0000277, -0.0014008, 0.0013179
1: -0.0043764, -0.0038097, -0.0043648, -0.0037875, -0.0004887, 0.0004718
2: 0.0122273, 0.0145641, 0.0122874, 0.0146350, -0.0018201, 0.0017084
3: 1.0079663, 1.0094035, 1.0079130, 1.0093747, -0.0014085, 0.0014905
4: -0.0039791, -0.0035958, -0.0039923, -0.0036048, -0.0002706, 0.0002888
5: 0.0025576, 0.0039000, 0.0025946, 0.0039363, -0.0010744, 0.0010104
6: -0.0024880, -0.0023510, -0.0024886, -0.0023534, -0.0001346, 0.0001376
7: -0.0129872, -0.0106944, -0.0129926, -0.0107986, -0.0021422, 0.0022528
8: -0.0105285, -0.0063707, -0.0106822, -0.0064625, -0.0028766, 0.0030741
9: -0.0011400, 0.0009225, -0.0010972, 0.0010046, -0.0014971, 0.0013968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008071, upper bound: 0.0008072
time: 0.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008071, upper bound: 0.0008108
time: 0.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0017672, -0.0000431, -0.0017674, -0.0000191, -0.0013812, 0.0013203
1: -0.0043648, -0.0037947, -0.0043648, -0.0037832, -0.0004906, 0.0004677
2: 0.0122875, 0.0146113, 0.0122873, 0.0146482, -0.0018059, 0.0017109
3: 1.0079296, 1.0093747, 1.0079025, 1.0093747, -0.0014452, 0.0014722
4: -0.0039879, -0.0036049, -0.0039948, -0.0036048, -0.0002710, 0.0002888
5: 0.0025947, 0.0039241, 0.0025945, 0.0039430, -0.0010603, 0.0010123
6: -0.0024878, -0.0023534, -0.0024891, -0.0023534, -0.0001345, 0.0001357
7: -0.0129908, -0.0107988, -0.0129936, -0.0107985, -0.0021443, 0.0021495
8: -0.0106309, -0.0064627, -0.0107108, -0.0064623, -0.0028815, 0.0030881
9: -0.0010971, 0.0009772, -0.0010973, 0.0010199, -0.0015100, 0.0014004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008108, upper bound: 0.0008073
time: 0.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008108, upper bound: 0.0008192
time: 0.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.08 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 3, lower bound: -0.0008071, upper bound: 0.0008072
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 3, lower bound: -0.0008071, upper bound: 0.0008108
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 3, lower bound: -0.0008108, upper bound: 0.0008073
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 3, lower bound: -0.0008108, upper bound: 0.0008192

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0018158, -0.0000739, -0.0018158, -0.0000739, -0.0013502, 0.0013502
1: -0.0043764, -0.0038097, -0.0043764, -0.0038097, -0.0004744, 0.0004744
2: 0.0122273, 0.0145641, 0.0122273, 0.0145641, -0.0017423, 0.0017423
3: 1.0079663, 1.0094035, 1.0079663, 1.0094035, -0.0014372, 0.0014372
4: -0.0039791, -0.0035958, -0.0039791, -0.0035958, -0.0002743, 0.0002743
5: 0.0025576, 0.0039000, 0.0025576, 0.0039000, -0.0010345, 0.0010345
6: -0.0024880, -0.0023510, -0.0024880, -0.0023510, -0.0001369, 0.0001369
7: -0.0129872, -0.0106944, -0.0129872, -0.0106944, -0.0022469, 0.0022469
8: -0.0105285, -0.0063707, -0.0105285, -0.0063707, -0.0029054, 0.0029054
9: -0.0011400, 0.0009225, -0.0011400, 0.0009225, -0.0014070, 0.0014070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007790, upper bound: 0.0007784
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007739, upper bound: 0.0007740
time: 0.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0018158, -0.0000739, -0.0017672, -0.0000431, -0.0014036, 0.0013176
1: -0.0043764, -0.0038097, -0.0043648, -0.0037947, -0.0004839, 0.0004717
2: 0.0122273, 0.0145641, 0.0122875, 0.0146113, -0.0018243, 0.0017081
3: 1.0079663, 1.0094035, 1.0079296, 1.0093747, -0.0014085, 0.0014739
4: -0.0039791, -0.0035958, -0.0039879, -0.0036049, -0.0002706, 0.0002896
5: 0.0025576, 0.0039000, 0.0025947, 0.0039241, -0.0010765, 0.0010102
6: -0.0024880, -0.0023510, -0.0024878, -0.0023534, -0.0001346, 0.0001368
7: -0.0129872, -0.0106944, -0.0129908, -0.0107988, -0.0021420, 0.0022531
8: -0.0105285, -0.0063707, -0.0106309, -0.0064627, -0.0028762, 0.0030832
9: -0.0011400, 0.0009225, -0.0010971, 0.0009772, -0.0015020, 0.0013967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007784, upper bound: 0.0007839
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007739, upper bound: 0.0007832
time: 0.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0017672, -0.0000431, -0.0018158, -0.0000739, -0.0013176, 0.0014036
1: -0.0043648, -0.0037947, -0.0043764, -0.0038097, -0.0004717, 0.0004839
2: 0.0122875, 0.0146113, 0.0122273, 0.0145641, -0.0017081, 0.0018243
3: 1.0079296, 1.0093747, 1.0079663, 1.0094035, -0.0014739, 0.0014085
4: -0.0039879, -0.0036049, -0.0039791, -0.0035958, -0.0002896, 0.0002706
5: 0.0025947, 0.0039241, 0.0025576, 0.0039000, -0.0010102, 0.0010765
6: -0.0024878, -0.0023534, -0.0024880, -0.0023510, -0.0001368, 0.0001346
7: -0.0129908, -0.0107988, -0.0129872, -0.0106944, -0.0022531, 0.0021420
8: -0.0106309, -0.0064627, -0.0105285, -0.0063707, -0.0030832, 0.0028762
9: -0.0010971, 0.0009772, -0.0011400, 0.0009225, -0.0013967, 0.0015020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007838, upper bound: 0.0007783
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007834, upper bound: 0.0007749
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0017672, -0.0000431, -0.0017672, -0.0000431, -0.0013199, 0.0013199
1: -0.0043648, -0.0037947, -0.0043648, -0.0037947, -0.0004677, 0.0004677
2: 0.0122875, 0.0146113, 0.0122875, 0.0146113, -0.0017103, 0.0017103
3: 1.0079296, 1.0093747, 1.0079296, 1.0093747, -0.0014452, 0.0014452
4: -0.0039879, -0.0036049, -0.0039879, -0.0036049, -0.0002709, 0.0002709
5: 0.0025947, 0.0039241, 0.0025947, 0.0039241, -0.0010119, 0.0010119
6: -0.0024878, -0.0023534, -0.0024878, -0.0023534, -0.0001344, 0.0001344
7: -0.0129908, -0.0107988, -0.0129908, -0.0107988, -0.0021441, 0.0021441
8: -0.0106309, -0.0064627, -0.0106309, -0.0064627, -0.0028807, 0.0028807
9: -0.0010971, 0.0009772, -0.0010971, 0.0009772, -0.0014003, 0.0014003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007838, upper bound: 0.0007992
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007834, upper bound: 0.0007978
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.47 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.47
Output dim: 3, lower bound: -0.0007790, upper bound: 0.0007784
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.47
Output dim: 3, lower bound: -0.0007739, upper bound: 0.0007740
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 3.47
Output dim: 3, lower bound: -0.0007784, upper bound: 0.0007839
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 3.47
Output dim: 3, lower bound: -0.0007739, upper bound: 0.0007832
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.47
Output dim: 3, lower bound: -0.0007838, upper bound: 0.0007783
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.47
Output dim: 3, lower bound: -0.0007834, upper bound: 0.0007749
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 3, lower bound: -0.0007838, upper bound: 0.0007992
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 3, lower bound: -0.0007834, upper bound: 0.0007978

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0017669, -0.0000616, -0.0017672, -0.0000431, -0.0013194, 0.0012999
1: -0.0043648, -0.0038035, -0.0043648, -0.0037947, -0.0004677, 0.0004589
2: 0.0122878, 0.0145830, 0.0122875, 0.0146113, -0.0017098, 0.0016796
3: 1.0079507, 1.0093747, 1.0079296, 1.0093747, -0.0014241, 0.0014452
4: -0.0039827, -0.0036049, -0.0039879, -0.0036049, -0.0002652, 0.0002708
5: 0.0025949, 0.0039096, 0.0025947, 0.0039241, -0.0010116, 0.0009962
6: -0.0024871, -0.0023534, -0.0024878, -0.0023534, -0.0001337, 0.0001344
7: -0.0129886, -0.0107992, -0.0129908, -0.0107988, -0.0021417, 0.0021437
8: -0.0105694, -0.0064633, -0.0106309, -0.0064627, -0.0028141, 0.0028798
9: -0.0010970, 0.0009443, -0.0010971, 0.0009772, -0.0014001, 0.0013647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007973
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007978
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0018605, -0.0000880, -0.0017671, -0.0000507, -0.0014138, 0.0013027
1: -0.0043850, -0.0038165, -0.0043648, -0.0037985, -0.0004901, 0.0004688
2: 0.0121739, 0.0145423, 0.0122877, 0.0145997, -0.0018268, 0.0016840
3: 1.0079775, 1.0094250, 1.0079383, 1.0093747, -0.0013973, 0.0014868
4: -0.0039751, -0.0035882, -0.0039858, -0.0036049, -0.0002660, 0.0002882
5: 0.0025237, 0.0038888, 0.0025948, 0.0039182, -0.0010834, 0.0009984
6: -0.0024888, -0.0023492, -0.0024875, -0.0023534, -0.0001354, 0.0001382
7: -0.0129855, -0.0106154, -0.0129899, -0.0107990, -0.0021419, 0.0023280
8: -0.0104813, -0.0062970, -0.0106057, -0.0064630, -0.0028238, 0.0030565
9: -0.0011719, 0.0008972, -0.0010971, 0.0009637, -0.0014835, 0.0013700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007641, upper bound: 0.0007801
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007641, upper bound: 0.0007624
time: 0.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.94 seconds
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007973
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 3, lower bound: -0.0007977, upper bound: 0.0007978
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 3, lower bound: -0.0007641, upper bound: 0.0007801
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.94
Output dim: 3, lower bound: -0.0007641, upper bound: 0.0007624

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017669, -0.0000616, -0.0017669, -0.0000616, -0.0012994, 0.0012994
1: -0.0043648, -0.0038035, -0.0043648, -0.0038035, -0.0004589, 0.0004589
2: 0.0122878, 0.0145830, 0.0122878, 0.0145830, -0.0016790, 0.0016790
3: 1.0079507, 1.0093747, 1.0079507, 1.0093747, -0.0014241, 0.0014241
4: -0.0039827, -0.0036049, -0.0039827, -0.0036049, -0.0002651, 0.0002651
5: 0.0025949, 0.0039096, 0.0025949, 0.0039096, -0.0009958, 0.0009958
6: -0.0024871, -0.0023534, -0.0024871, -0.0023534, -0.0001337, 0.0001337
7: -0.0129886, -0.0107992, -0.0129886, -0.0107992, -0.0021413, 0.0021413
8: -0.0105694, -0.0064633, -0.0105694, -0.0064633, -0.0028132, 0.0028132
9: -0.0010970, 0.0009443, -0.0010970, 0.0009443, -0.0013645, 0.0013645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007827, upper bound: 0.0007664
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007668, upper bound: 0.0007663
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017669, -0.0000616, -0.0018605, -0.0000880, -0.0012957, 0.0013997
1: -0.0043648, -0.0038035, -0.0043850, -0.0038165, -0.0004565, 0.0004845
2: 0.0122878, 0.0145830, 0.0121739, 0.0145423, -0.0016734, 0.0018051
3: 1.0079507, 1.0093747, 1.0079775, 1.0094250, -0.0014744, 0.0013973
4: -0.0039827, -0.0036049, -0.0039751, -0.0035882, -0.0002841, 0.0002640
5: 0.0025949, 0.0039096, 0.0025237, 0.0038888, -0.0009929, 0.0010723
6: -0.0024871, -0.0023534, -0.0024888, -0.0023492, -0.0001378, 0.0001354
7: -0.0129886, -0.0107992, -0.0129855, -0.0106154, -0.0023264, 0.0021409
8: -0.0105694, -0.0064633, -0.0104813, -0.0062970, -0.0030095, 0.0028009
9: -0.0010970, 0.0009443, -0.0011719, 0.0008972, -0.0013580, 0.0014584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007827, upper bound: 0.0007664
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0007668, upper bound: 0.0007665
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.93 seconds
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 3, lower bound: -0.0007827, upper bound: 0.0007664
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 3, lower bound: -0.0007668, upper bound: 0.0007663
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 3, lower bound: -0.0007827, upper bound: 0.0007664
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 3, lower bound: -0.0007668, upper bound: 0.0007665

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.02 + 31.99 = 35.01 seconds
