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
Threshold: 0.00073336


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0071140, 0.0082858, 0.0071140, 0.0082858, -0.0005638, 0.0005638)
1: (0.0023501, 0.0025194, 0.0023501, 0.0025194, -0.0000814, 0.0000814)
2: (0.0097789, 0.0104267, 0.0097789, 0.0104267, -0.0003117, 0.0003117)
3: (-0.0045667, -0.0038967, -0.0045667, -0.0038967, -0.0003224, 0.0003224)
4: (0.0001814, 0.0009067, 0.0001814, 0.0009067, -0.0003490, 0.0003490)
5: (0.0032553, 0.0039417, 0.0032553, 0.0039417, -0.0003303, 0.0003303)
6: (-0.0093844, -0.0066609, -0.0093844, -0.0066609, -0.0013103, 0.0013103)
7: (0.0065148, 0.0102240, 0.0065148, 0.0102240, -0.0017846, 0.0017846)
8: (0.9938030, 0.9964159, 0.9938030, 0.9964159, -0.0012571, 0.0012571)
9: (-0.0126339, -0.0102621, -0.0126339, -0.0102621, -0.0011411, 0.0011411)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.70 + 1.36 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0009348, upper bound: 0.0009348

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007788, upper bound: 0.0008750
time: 0.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708
time: 0.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 8, lower bound: -0.0007788, upper bound: 0.0008750
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0071267, 0.0082853, 0.0071140, 0.0082858, -0.0005514, 0.0005634
1: 0.0023519, 0.0025193, 0.0023501, 0.0025194, -0.0000797, 0.0000814
2: 0.0097791, 0.0104197, 0.0097789, 0.0104267, -0.0003115, 0.0003048
3: -0.0045664, -0.0039039, -0.0045667, -0.0038967, -0.0003222, 0.0003153
4: 0.0001893, 0.0009065, 0.0001814, 0.0009067, -0.0003413, 0.0003488
5: 0.0032555, 0.0039342, 0.0032553, 0.0039417, -0.0003301, 0.0003230
6: -0.0093833, -0.0066904, -0.0093844, -0.0066609, -0.0013095, 0.0012815
7: 0.0065550, 0.0102226, 0.0065148, 0.0102240, -0.0017453, 0.0017835
8: 0.9938313, 0.9964148, 0.9938030, 0.9964159, -0.0012294, 0.0012563
9: -0.0126329, -0.0102878, -0.0126339, -0.0102621, -0.0011404, 0.0011160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708
time: 0.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708
time: 0.47 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0071856, 0.0084411, 0.0071265, 0.0082848, -0.0005830, 0.0007575
1: 0.0023604, 0.0025418, 0.0023519, 0.0025192, -0.0000842, 0.0001094
2: 0.0096930, 0.0103871, 0.0097794, 0.0104198, -0.0004188, 0.0003223
3: -0.0046555, -0.0039376, -0.0045661, -0.0039038, -0.0004332, 0.0003334
4: 0.0002257, 0.0010029, 0.0001891, 0.0009061, -0.0003609, 0.0004689
5: 0.0031643, 0.0038997, 0.0032558, 0.0039344, -0.0004438, 0.0003415
6: -0.0097454, -0.0068273, -0.0093821, -0.0066900, -0.0017607, 0.0013551
7: 0.0067414, 0.0107157, 0.0065544, 0.0102210, -0.0018455, 0.0023979
8: 0.9939628, 0.9967623, 0.9938309, 0.9964137, -0.0013000, 0.0016891
9: -0.0129483, -0.0104070, -0.0126319, -0.0102874, -0.0015333, 0.0011801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708
time: 0.48 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708
time: 0.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 8, lower bound: -0.0007708, upper bound: 0.0007708

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0071267, 0.0082853, 0.0071267, 0.0082853, -0.0005510, 0.0005510
1: 0.0023519, 0.0025193, 0.0023519, 0.0025193, -0.0000796, 0.0000796
2: 0.0097791, 0.0104197, 0.0097791, 0.0104197, -0.0003046, 0.0003046
3: -0.0045664, -0.0039039, -0.0045664, -0.0039039, -0.0003151, 0.0003151
4: 0.0001893, 0.0009065, 0.0001893, 0.0009065, -0.0003411, 0.0003411
5: 0.0032555, 0.0039342, 0.0032555, 0.0039342, -0.0003228, 0.0003228
6: -0.0093833, -0.0066904, -0.0093833, -0.0066904, -0.0012807, 0.0012807
7: 0.0065550, 0.0102226, 0.0065550, 0.0102226, -0.0017442, 0.0017442
8: 0.9938313, 0.9964148, 0.9938313, 0.9964148, -0.0012286, 0.0012286
9: -0.0126329, -0.0102878, -0.0126329, -0.0102878, -0.0011153, 0.0011153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003305, upper bound: 0.0003660
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007715, upper bound: 0.0008680
time: 0.48 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0071267, 0.0082853, 0.0071856, 0.0084411, -0.0007540, 0.0005335
1: 0.0023519, 0.0025193, 0.0023604, 0.0025418, -0.0001089, 0.0000771
2: 0.0097791, 0.0104197, 0.0096930, 0.0103871, -0.0002949, 0.0004168
3: -0.0045664, -0.0039039, -0.0046555, -0.0039376, -0.0003050, 0.0004311
4: 0.0001893, 0.0009065, 0.0002257, 0.0010029, -0.0004667, 0.0003302
5: 0.0032555, 0.0039342, 0.0031643, 0.0038997, -0.0003125, 0.0004417
6: -0.0093833, -0.0066904, -0.0097454, -0.0068273, -0.0012399, 0.0017524
7: 0.0065550, 0.0102226, 0.0067414, 0.0107157, -0.0023867, 0.0016887
8: 0.9938313, 0.9964148, 0.9939628, 0.9967623, -0.0016812, 0.0011896
9: -0.0126329, -0.0102878, -0.0129483, -0.0104070, -0.0010798, 0.0015261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003305, upper bound: 0.0003660
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007715, upper bound: 0.0008680
time: 0.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0071856, 0.0084411, 0.0071267, 0.0082853, -0.0005335, 0.0007540
1: 0.0023604, 0.0025418, 0.0023519, 0.0025193, -0.0000771, 0.0001089
2: 0.0096930, 0.0103871, 0.0097791, 0.0104197, -0.0004168, 0.0002949
3: -0.0046555, -0.0039376, -0.0045664, -0.0039039, -0.0004311, 0.0003050
4: 0.0002257, 0.0010029, 0.0001893, 0.0009065, -0.0003302, 0.0004667
5: 0.0031643, 0.0038997, 0.0032555, 0.0039342, -0.0004417, 0.0003125
6: -0.0097454, -0.0068273, -0.0093833, -0.0066904, -0.0017524, 0.0012399
7: 0.0067414, 0.0107157, 0.0065550, 0.0102226, -0.0016887, 0.0023867
8: 0.9939628, 0.9967623, 0.9938313, 0.9964148, -0.0011896, 0.0016812
9: -0.0129483, -0.0104070, -0.0126329, -0.0102878, -0.0015261, 0.0010798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005228, upper bound: 0.0005105
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007704, upper bound: 0.0007704
time: 0.48 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0071856, 0.0084411, 0.0071856, 0.0084411, -0.0005869, 0.0005869
1: 0.0023604, 0.0025418, 0.0023604, 0.0025418, -0.0000848, 0.0000848
2: 0.0096930, 0.0103871, 0.0096930, 0.0103871, -0.0003245, 0.0003245
3: -0.0046555, -0.0039376, -0.0046555, -0.0039376, -0.0003356, 0.0003356
4: 0.0002257, 0.0010029, 0.0002257, 0.0010029, -0.0003633, 0.0003633
5: 0.0031643, 0.0038997, 0.0031643, 0.0038997, -0.0003438, 0.0003438
6: -0.0097454, -0.0068273, -0.0097454, -0.0068273, -0.0013642, 0.0013642
7: 0.0067414, 0.0107157, 0.0067414, 0.0107157, -0.0018579, 0.0018579
8: 0.9939628, 0.9967623, 0.9939628, 0.9967623, -0.0013087, 0.0013087
9: -0.0129483, -0.0104070, -0.0129483, -0.0104070, -0.0011880, 0.0011880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005228, upper bound: 0.0005105
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007704, upper bound: 0.0007704
time: 0.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 10.28 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 10.28
Output dim: 8, lower bound: -0.0003305, upper bound: 0.0003660
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.28
Output dim: 8, lower bound: -0.0007715, upper bound: 0.0008680
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 10.28
Output dim: 8, lower bound: -0.0003305, upper bound: 0.0003660
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.28
Output dim: 8, lower bound: -0.0007715, upper bound: 0.0008680
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 10.28
Output dim: 8, lower bound: -0.0005228, upper bound: 0.0005105
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.28
Output dim: 8, lower bound: -0.0007704, upper bound: 0.0007704
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 10.28
Output dim: 8, lower bound: -0.0005228, upper bound: 0.0005105
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.28
Output dim: 8, lower bound: -0.0007704, upper bound: 0.0007704

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0071269, 0.0082834, 0.0071267, 0.0082853, -0.0005509, 0.0005102
1: 0.0023519, 0.0025190, 0.0023519, 0.0025193, -0.0000796, 0.0000737
2: 0.0097802, 0.0104196, 0.0097791, 0.0104197, -0.0002821, 0.0003046
3: -0.0045654, -0.0039040, -0.0045664, -0.0039039, -0.0002917, 0.0003150
4: 0.0001894, 0.0009053, 0.0001893, 0.0009065, -0.0003410, 0.0003158
5: 0.0032566, 0.0039341, 0.0032555, 0.0039342, -0.0002989, 0.0003227
6: -0.0093790, -0.0066909, -0.0093833, -0.0066904, -0.0011859, 0.0012805
7: 0.0065558, 0.0102166, 0.0065550, 0.0102226, -0.0017439, 0.0016150
8: 0.9938318, 0.9964107, 0.9938313, 0.9964148, -0.0012285, 0.0011377
9: -0.0126292, -0.0102883, -0.0126329, -0.0102878, -0.0010327, 0.0011151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007947, upper bound: 0.0008139
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0071269, 0.0082834, 0.0071856, 0.0084411, -0.0007539, 0.0004966
1: 0.0023519, 0.0025190, 0.0023604, 0.0025418, -0.0001089, 0.0000717
2: 0.0097802, 0.0104196, 0.0096930, 0.0103871, -0.0002746, 0.0004168
3: -0.0045654, -0.0039040, -0.0046555, -0.0039376, -0.0002840, 0.0004311
4: 0.0001894, 0.0009053, 0.0002257, 0.0010029, -0.0004667, 0.0003074
5: 0.0032566, 0.0039341, 0.0031643, 0.0038997, -0.0002909, 0.0004416
6: -0.0093790, -0.0066909, -0.0097454, -0.0068273, -0.0011543, 0.0017522
7: 0.0065558, 0.0102166, 0.0067414, 0.0107157, -0.0023864, 0.0015720
8: 0.9938318, 0.9964107, 0.9939628, 0.9967623, -0.0016810, 0.0011074
9: -0.0126292, -0.0102883, -0.0129483, -0.0104070, -0.0010052, 0.0015259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004745, upper bound: 0.0006545
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005517, upper bound: 0.0007430
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007711, upper bound: 0.0008676
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0071864, 0.0084403, 0.0071267, 0.0082853, -0.0005309, 0.0007530
1: 0.0023605, 0.0025417, 0.0023519, 0.0025193, -0.0000767, 0.0001088
2: 0.0096934, 0.0103867, 0.0097791, 0.0104197, -0.0004163, 0.0002935
3: -0.0046551, -0.0039381, -0.0045664, -0.0039039, -0.0004306, 0.0003036
4: 0.0002262, 0.0010024, 0.0001893, 0.0009065, -0.0003286, 0.0004661
5: 0.0031647, 0.0038992, 0.0032555, 0.0039342, -0.0004411, 0.0003110
6: -0.0097436, -0.0068293, -0.0093833, -0.0066904, -0.0017502, 0.0012339
7: 0.0067442, 0.0107132, 0.0065550, 0.0102226, -0.0016805, 0.0023837
8: 0.9939647, 0.9967605, 0.9938313, 0.9964148, -0.0011838, 0.0016791
9: -0.0129467, -0.0104087, -0.0126329, -0.0102878, -0.0015242, 0.0010746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003656, upper bound: 0.0003300
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008676, upper bound: 0.0007711
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0071864, 0.0084403, 0.0071856, 0.0084411, -0.0005847, 0.0005879
1: 0.0023605, 0.0025417, 0.0023604, 0.0025418, -0.0000845, 0.0000849
2: 0.0096934, 0.0103867, 0.0096930, 0.0103871, -0.0003250, 0.0003233
3: -0.0046551, -0.0039381, -0.0046555, -0.0039376, -0.0003361, 0.0003343
4: 0.0002262, 0.0010024, 0.0002257, 0.0010029, -0.0003620, 0.0003639
5: 0.0031647, 0.0038992, 0.0031643, 0.0038997, -0.0003444, 0.0003425
6: -0.0097436, -0.0068293, -0.0097454, -0.0068273, -0.0013663, 0.0013591
7: 0.0067442, 0.0107132, 0.0067414, 0.0107157, -0.0018509, 0.0018608
8: 0.9939647, 0.9967605, 0.9939628, 0.9967623, -0.0013038, 0.0013108
9: -0.0129467, -0.0104087, -0.0129483, -0.0104070, -0.0011899, 0.0011835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005105, upper bound: 0.0005228
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005105, upper bound: 0.0007704
time: 0.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 10.06 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.06
Output dim: 8, lower bound: -0.0007947, upper bound: 0.0008139
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.06
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.06
Output dim: 8, lower bound: -0.0005517, upper bound: 0.0007430
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.06
Output dim: 8, lower bound: -0.0007711, upper bound: 0.0008676
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 10.06
Output dim: 8, lower bound: -0.0003656, upper bound: 0.0003300
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.06
Output dim: 8, lower bound: -0.0008676, upper bound: 0.0007711
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 10.06
Output dim: 8, lower bound: -0.0005105, upper bound: 0.0005228
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.06
Output dim: 8, lower bound: -0.0005105, upper bound: 0.0007704

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0071269, 0.0082834, 0.0071270, 0.0082754, -0.0005401, 0.0005099
1: 0.0023519, 0.0025190, 0.0023519, 0.0025179, -0.0000780, 0.0000737
2: 0.0097802, 0.0104196, 0.0097846, 0.0104195, -0.0002819, 0.0002986
3: -0.0045654, -0.0039040, -0.0045607, -0.0039041, -0.0002916, 0.0003088
4: 0.0001894, 0.0009053, 0.0001894, 0.0009003, -0.0003343, 0.0003157
5: 0.0032566, 0.0039341, 0.0032614, 0.0039341, -0.0002987, 0.0003164
6: -0.0093790, -0.0066909, -0.0093602, -0.0066910, -0.0011852, 0.0012553
7: 0.0065558, 0.0102166, 0.0065559, 0.0101911, -0.0017096, 0.0016142
8: 0.9938318, 0.9964107, 0.9938319, 0.9963926, -0.0012043, 0.0011370
9: -0.0126292, -0.0102883, -0.0126128, -0.0102884, -0.0010321, 0.0010932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0071275, 0.0082672, 0.0069811, 0.0082055, -0.0005496, 0.0007009
1: 0.0023520, 0.0025167, 0.0023309, 0.0025078, -0.0000794, 0.0001013
2: 0.0097892, 0.0104193, 0.0098233, 0.0105002, -0.0003875, 0.0003038
3: -0.0045560, -0.0039044, -0.0045208, -0.0038207, -0.0004008, 0.0003142
4: 0.0001898, 0.0008952, 0.0000991, 0.0008571, -0.0003402, 0.0004339
5: 0.0032662, 0.0039338, 0.0033023, 0.0040195, -0.0004106, 0.0003219
6: -0.0093412, -0.0066923, -0.0091978, -0.0063519, -0.0016291, 0.0012774
7: 0.0065576, 0.0101651, 0.0060941, 0.0099699, -0.0017396, 0.0022186
8: 0.9938331, 0.9963744, 0.9935067, 0.9962369, -0.0012254, 0.0015629
9: -0.0125962, -0.0102894, -0.0124714, -0.0099931, -0.0014187, 0.0011124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0071278, 0.0082824, 0.0071868, 0.0084423, -0.0007505, 0.0004930
1: 0.0023521, 0.0025189, 0.0023606, 0.0025420, -0.0001084, 0.0000712
2: 0.0097808, 0.0104191, 0.0096923, 0.0103865, -0.0002725, 0.0004149
3: -0.0045647, -0.0039045, -0.0046562, -0.0039383, -0.0002819, 0.0004292
4: 0.0001899, 0.0009046, 0.0002265, 0.0010036, -0.0004646, 0.0003051
5: 0.0032573, 0.0039336, 0.0031636, 0.0038990, -0.0002888, 0.0004397
6: -0.0093765, -0.0066929, -0.0097482, -0.0068301, -0.0011458, 0.0017444
7: 0.0065584, 0.0102132, 0.0067452, 0.0107195, -0.0023757, 0.0015604
8: 0.9938337, 0.9964083, 0.9939653, 0.9967648, -0.0016735, 0.0010992
9: -0.0126270, -0.0102900, -0.0129507, -0.0104094, -0.0009978, 0.0015191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0006647
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0007430
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0071269, 0.0082834, 0.0071864, 0.0084403, -0.0007529, 0.0004941
1: 0.0023519, 0.0025190, 0.0023605, 0.0025417, -0.0001088, 0.0000714
2: 0.0097802, 0.0104196, 0.0096934, 0.0103867, -0.0002732, 0.0004163
3: -0.0045654, -0.0039040, -0.0046551, -0.0039381, -0.0002825, 0.0004305
4: 0.0001894, 0.0009053, 0.0002262, 0.0010024, -0.0004661, 0.0003059
5: 0.0032566, 0.0039341, 0.0031647, 0.0038992, -0.0002894, 0.0004411
6: -0.0093790, -0.0066909, -0.0097436, -0.0068293, -0.0011484, 0.0017500
7: 0.0065558, 0.0102166, 0.0067442, 0.0107132, -0.0023834, 0.0015641
8: 0.9938318, 0.9964107, 0.9939647, 0.9967605, -0.0016789, 0.0011018
9: -0.0126292, -0.0102883, -0.0129467, -0.0104087, -0.0010001, 0.0015240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006108, upper bound: 0.0007125
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006108, upper bound: 0.0008676
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0071864, 0.0084403, 0.0071269, 0.0082834, -0.0004941, 0.0007529
1: 0.0023605, 0.0025417, 0.0023519, 0.0025190, -0.0000714, 0.0001088
2: 0.0096934, 0.0103867, 0.0097802, 0.0104196, -0.0004163, 0.0002732
3: -0.0046551, -0.0039381, -0.0045654, -0.0039040, -0.0004305, 0.0002825
4: 0.0002262, 0.0010024, 0.0001894, 0.0009053, -0.0003059, 0.0004661
5: 0.0031647, 0.0038992, 0.0032566, 0.0039341, -0.0004411, 0.0002894
6: -0.0097436, -0.0068293, -0.0093790, -0.0066909, -0.0017500, 0.0011484
7: 0.0067442, 0.0107132, 0.0065558, 0.0102166, -0.0015641, 0.0023834
8: 0.9939647, 0.9967605, 0.9938318, 0.9964107, -0.0011018, 0.0016789
9: -0.0129467, -0.0104087, -0.0126292, -0.0102883, -0.0015240, 0.0010001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006540, upper bound: 0.0004741
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 143
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 96

Time for candidate selection: 8.71 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007956, upper bound: 0.0005758
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007188, upper bound: 0.0005631
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0071864, 0.0084403, 0.0071864, 0.0084403, -0.0005858, 0.0005858
1: 0.0023605, 0.0025417, 0.0023605, 0.0025417, -0.0000846, 0.0000846
2: 0.0096934, 0.0103867, 0.0096934, 0.0103867, -0.0003239, 0.0003239
3: -0.0046551, -0.0039381, -0.0046551, -0.0039381, -0.0003350, 0.0003350
4: 0.0002262, 0.0010024, 0.0002262, 0.0010024, -0.0003626, 0.0003626
5: 0.0031647, 0.0038992, 0.0031647, 0.0038992, -0.0003432, 0.0003432
6: -0.0097436, -0.0068293, -0.0097436, -0.0068293, -0.0013615, 0.0013615
7: 0.0067442, 0.0107132, 0.0067442, 0.0107132, -0.0018543, 0.0018543
8: 0.9939647, 0.9967605, 0.9939647, 0.9967605, -0.0013062, 0.0013062
9: -0.0129467, -0.0104087, -0.0129467, -0.0104087, -0.0011857, 0.0011857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 143
type: A, layer: 3, pos: 156
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 96

Time for candidate selection: 7.93 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 62

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 225

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003750, upper bound: 0.0006154
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004913, upper bound: 0.0007533
time: 0.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.20 seconds
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0007619, upper bound: 0.0007619
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0006647
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0007430
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0006108, upper bound: 0.0007125
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0006108, upper bound: 0.0008676
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0007956, upper bound: 0.0005758
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0007188, upper bound: 0.0005631
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0003750, upper bound: 0.0006154
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 8, lower bound: -0.0004913, upper bound: 0.0007533

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0071272, 0.0082735, 0.0071270, 0.0082754, -0.0005398, 0.0004986
1: 0.0023520, 0.0025176, 0.0023519, 0.0025179, -0.0000780, 0.0000720
2: 0.0097857, 0.0104194, 0.0097846, 0.0104195, -0.0002756, 0.0002984
3: -0.0045597, -0.0039042, -0.0045607, -0.0039041, -0.0002851, 0.0003087
4: 0.0001896, 0.0008992, 0.0001894, 0.0009003, -0.0003341, 0.0003086
5: 0.0032624, 0.0039339, 0.0032614, 0.0039341, -0.0002921, 0.0003162
6: -0.0093559, -0.0066916, -0.0093602, -0.0066910, -0.0011588, 0.0012546
7: 0.0065566, 0.0101852, 0.0065559, 0.0101911, -0.0017087, 0.0015782
8: 0.9938325, 0.9963886, 0.9938319, 0.9963926, -0.0012037, 0.0011117
9: -0.0126091, -0.0102888, -0.0126128, -0.0102884, -0.0010092, 0.0010926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006089, upper bound: 0.0007269
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007942, upper bound: 0.0008134
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069814, 0.0082033, 0.0071270, 0.0082754, -0.0007282, 0.0004818
1: 0.0023309, 0.0025074, 0.0023519, 0.0025179, -0.0001052, 0.0000696
2: 0.0098245, 0.0105001, 0.0097846, 0.0104195, -0.0002664, 0.0004026
3: -0.0045195, -0.0038208, -0.0045607, -0.0039041, -0.0002755, 0.0004164
4: 0.0000993, 0.0008557, 0.0001894, 0.0009003, -0.0004508, 0.0002982
5: 0.0033036, 0.0040194, 0.0032614, 0.0039341, -0.0002822, 0.0004266
6: -0.0091928, -0.0063526, -0.0093602, -0.0066910, -0.0011198, 0.0016925
7: 0.0060950, 0.0099630, 0.0065559, 0.0101911, -0.0023051, 0.0015250
8: 0.9935073, 0.9962320, 0.9938319, 0.9963926, -0.0016238, 0.0010743
9: -0.0124670, -0.0099936, -0.0126128, -0.0102884, -0.0009752, 0.0014739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006089, upper bound: 0.0007269
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007942, upper bound: 0.0008134
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0071272, 0.0082735, 0.0069811, 0.0082055, -0.0005121, 0.0007006
1: 0.0023520, 0.0025176, 0.0023309, 0.0025078, -0.0000740, 0.0001012
2: 0.0097857, 0.0104194, 0.0098233, 0.0105002, -0.0003874, 0.0002831
3: -0.0045597, -0.0039042, -0.0045208, -0.0038207, -0.0004006, 0.0002928
4: 0.0001896, 0.0008992, 0.0000991, 0.0008571, -0.0003170, 0.0004337
5: 0.0032624, 0.0039339, 0.0033023, 0.0040195, -0.0004104, 0.0003000
6: -0.0093559, -0.0066916, -0.0091978, -0.0063519, -0.0016284, 0.0011903
7: 0.0065566, 0.0101852, 0.0060941, 0.0099699, -0.0016211, 0.0022178
8: 0.9938325, 0.9963886, 0.9935067, 0.9962369, -0.0011420, 0.0015622
9: -0.0126091, -0.0102888, -0.0124714, -0.0099931, -0.0014181, 0.0010366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005125, upper bound: 0.0005883
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007614, upper bound: 0.0007614
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069814, 0.0082033, 0.0069811, 0.0082055, -0.0005693, 0.0005283
1: 0.0023309, 0.0025074, 0.0023309, 0.0025078, -0.0000823, 0.0000763
2: 0.0098245, 0.0105001, 0.0098233, 0.0105002, -0.0002921, 0.0003148
3: -0.0045195, -0.0038208, -0.0045208, -0.0038207, -0.0003021, 0.0003256
4: 0.0000993, 0.0008557, 0.0000991, 0.0008571, -0.0003524, 0.0003270
5: 0.0033036, 0.0040194, 0.0033023, 0.0040195, -0.0003095, 0.0003335
6: -0.0091928, -0.0063526, -0.0091978, -0.0063519, -0.0012279, 0.0013233
7: 0.0060950, 0.0099630, 0.0060941, 0.0099699, -0.0018022, 0.0016723
8: 0.9935073, 0.9962320, 0.9935067, 0.9962369, -0.0012695, 0.0011780
9: -0.0124670, -0.0099936, -0.0124714, -0.0099931, -0.0010693, 0.0011524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005125, upper bound: 0.0005883
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007614, upper bound: 0.0007614
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0071279, 0.0082826, 0.0071868, 0.0084423, -0.0007510, 0.0004922
1: 0.0023521, 0.0025189, 0.0023606, 0.0025420, -0.0001085, 0.0000711
2: 0.0097806, 0.0104191, 0.0096923, 0.0103865, -0.0002721, 0.0004152
3: -0.0045649, -0.0039046, -0.0046562, -0.0039383, -0.0002815, 0.0004295
4: 0.0001900, 0.0009048, 0.0002265, 0.0010036, -0.0004649, 0.0003047
5: 0.0032571, 0.0039336, 0.0031636, 0.0038990, -0.0002883, 0.0004400
6: -0.0093771, -0.0066931, -0.0097482, -0.0068301, -0.0011440, 0.0017456
7: 0.0065587, 0.0102141, 0.0067452, 0.0107195, -0.0023774, 0.0015581
8: 0.9938340, 0.9964090, 0.9939653, 0.9967648, -0.0016747, 0.0010975
9: -0.0126275, -0.0102902, -0.0129507, -0.0104094, -0.0009963, 0.0015202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 143
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 156
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 96

Time for candidate selection: 8.05 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0003054, upper bound: 0.0006568
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004250, upper bound: 0.0006994
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 225

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 143

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 62

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 156

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0007430
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5

No IS candidates found

### IS candidates at layer 7

No IS candidates found

No IS candidates found

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.05 + 188.99 = 192.04 seconds
