## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0010557


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0006020, 0.0014273, 0.0006020, 0.0014273, -0.0008165, 0.0008165)
1: (0.9928627, 0.9949434, 0.9928627, 0.9949434, -0.0019735, 0.0019735)
2: (-0.0071335, -0.0045059, -0.0071335, -0.0045059, -0.0026276, 0.0026276)
3: (0.0034410, 0.0043042, 0.0034410, 0.0043042, -0.0007896, 0.0007896)
4: (0.0023270, 0.0040549, 0.0023270, 0.0040549, -0.0017279, 0.0017279)
5: (0.0051665, 0.0071891, 0.0051665, 0.0071891, -0.0020225, 0.0020225)
6: (-0.0015972, -0.0006634, -0.0015972, -0.0006634, -0.0009337, 0.0009337)
7: (-0.0087886, -0.0073323, -0.0087886, -0.0073323, -0.0014563, 0.0014563)
8: (0.0030193, 0.0076241, 0.0030193, 0.0076241, -0.0040995, 0.0040995)
9: (-0.0047494, -0.0021082, -0.0047494, -0.0021082, -0.0026412, 0.0026412)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 1.79 = 3.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0012041, upper bound: 0.0012041

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011160, upper bound: 0.0011762
time: 1.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011825, upper bound: 0.0011824
time: 1.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.72
Output dim: 1, lower bound: -0.0011160, upper bound: 0.0011762
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.72
Output dim: 1, lower bound: -0.0011825, upper bound: 0.0011824

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0005984, 0.0013435, 0.0006038, 0.0014135, -0.0008061, 0.0007309
1: 0.9930739, 0.9949527, 0.9928973, 0.9949393, -0.0017560, 0.0019447
2: -0.0067658, -0.0044973, -0.0070730, -0.0045099, -0.0022559, 0.0025756
3: 0.0034372, 0.0042165, 0.0034428, 0.0042897, -0.0007772, 0.0006992
4: 0.0023228, 0.0037643, 0.0023290, 0.0040071, -0.0016843, 0.0014354
5: 0.0051576, 0.0069836, 0.0051706, 0.0071552, -0.0019976, 0.0018130
6: -0.0014696, -0.0006603, -0.0015762, -0.0006649, -0.0008047, 0.0009159
7: -0.0086407, -0.0073259, -0.0087643, -0.0073353, -0.0013054, 0.0014384
8: 0.0029996, 0.0071411, 0.0030286, 0.0075446, -0.0040359, 0.0036332
9: -0.0044811, -0.0020966, -0.0047052, -0.0021136, -0.0023675, 0.0026086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011160, upper bound: 0.0011160
time: 1.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011160, upper bound: 0.0011762
time: 1.13 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0006034, 0.0014184, 0.0006020, 0.0014273, -0.0008152, 0.0008069
1: 0.9928850, 0.9949400, 0.9928627, 0.9949434, -0.0019419, 0.0019702
2: -0.0070944, -0.0045091, -0.0071335, -0.0045059, -0.0025885, 0.0026244
3: 0.0034424, 0.0042948, 0.0034410, 0.0043042, -0.0007883, 0.0007740
4: 0.0023286, 0.0040240, 0.0023270, 0.0040549, -0.0017264, 0.0016970
5: 0.0051698, 0.0071672, 0.0051665, 0.0071891, -0.0020192, 0.0020007
6: -0.0015836, -0.0006646, -0.0015972, -0.0006634, -0.0009202, 0.0009326
7: -0.0087729, -0.0073347, -0.0087886, -0.0073323, -0.0014406, 0.0014540
8: 0.0030267, 0.0075728, 0.0030193, 0.0076241, -0.0040948, 0.0039106
9: -0.0047209, -0.0021125, -0.0047494, -0.0021082, -0.0026126, 0.0026368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011762, upper bound: 0.0011160
time: 1.04 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011762, upper bound: 0.0011824
time: 0.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 1, lower bound: -0.0011160, upper bound: 0.0011160
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 1, lower bound: -0.0011160, upper bound: 0.0011762
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 1, lower bound: -0.0011762, upper bound: 0.0011160
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 1, lower bound: -0.0011762, upper bound: 0.0011824

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0005984, 0.0013435, 0.0005984, 0.0013435, -0.0007360, 0.0007360
1: 0.9930739, 0.9949527, 0.9930739, 0.9949527, -0.0017664, 0.0017664
2: -0.0067658, -0.0044973, -0.0067658, -0.0044973, -0.0022685, 0.0022685
3: 0.0034372, 0.0042165, 0.0034372, 0.0042165, -0.0007030, 0.0007030
4: 0.0023228, 0.0037643, 0.0023228, 0.0037643, -0.0014415, 0.0014415
5: 0.0051576, 0.0069836, 0.0051576, 0.0069836, -0.0018260, 0.0018260
6: -0.0014696, -0.0006603, -0.0014696, -0.0006603, -0.0008093, 0.0008093
7: -0.0086407, -0.0073259, -0.0086407, -0.0073259, -0.0013148, 0.0013148
8: 0.0029996, 0.0071411, 0.0029996, 0.0071411, -0.0036555, 0.0036555
9: -0.0044811, -0.0020966, -0.0044811, -0.0020966, -0.0023845, 0.0023845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010854, upper bound: 0.0010758
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010973, upper bound: 0.0010987
time: 0.73 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0005984, 0.0013435, 0.0006034, 0.0014184, -0.0008110, 0.0007312
1: 0.9930739, 0.9949527, 0.9928850, 0.9949400, -0.0017569, 0.0019574
2: -0.0067658, -0.0044973, -0.0070944, -0.0045091, -0.0022567, 0.0025971
3: 0.0034372, 0.0042165, 0.0034424, 0.0042948, -0.0007829, 0.0006996
4: 0.0023228, 0.0037643, 0.0023286, 0.0040240, -0.0017013, 0.0014358
5: 0.0051576, 0.0069836, 0.0051698, 0.0071672, -0.0020096, 0.0018138
6: -0.0014696, -0.0006603, -0.0015836, -0.0006646, -0.0008050, 0.0009233
7: -0.0086407, -0.0073259, -0.0087729, -0.0073347, -0.0013060, 0.0014470
8: 0.0029996, 0.0071411, 0.0030267, 0.0075728, -0.0040699, 0.0036347
9: -0.0044811, -0.0020966, -0.0047209, -0.0021125, -0.0023686, 0.0026242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010854, upper bound: 0.0011332
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010973, upper bound: 0.0011573
time: 0.99 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006034, 0.0014184, 0.0005984, 0.0013435, -0.0007312, 0.0008110
1: 0.9928850, 0.9949400, 0.9930739, 0.9949527, -0.0019574, 0.0017569
2: -0.0070944, -0.0045091, -0.0067658, -0.0044973, -0.0025971, 0.0022567
3: 0.0034424, 0.0042948, 0.0034372, 0.0042165, -0.0006996, 0.0007829
4: 0.0023286, 0.0040240, 0.0023228, 0.0037643, -0.0014358, 0.0017013
5: 0.0051698, 0.0071672, 0.0051576, 0.0069836, -0.0018138, 0.0020096
6: -0.0015836, -0.0006646, -0.0014696, -0.0006603, -0.0009233, 0.0008050
7: -0.0087729, -0.0073347, -0.0086407, -0.0073259, -0.0014470, 0.0013060
8: 0.0030267, 0.0075728, 0.0029996, 0.0071411, -0.0036347, 0.0040699
9: -0.0047209, -0.0021125, -0.0044811, -0.0020966, -0.0026242, 0.0023686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011332, upper bound: 0.0010854
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011573, upper bound: 0.0010973
time: 1.04 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006034, 0.0014184, 0.0006034, 0.0014184, -0.0008056, 0.0008056
1: 0.9928850, 0.9949400, 0.9928850, 0.9949400, -0.0019386, 0.0019386
2: -0.0070944, -0.0045091, -0.0070944, -0.0045091, -0.0025853, 0.0025853
3: 0.0034424, 0.0042948, 0.0034424, 0.0042948, -0.0007727, 0.0007727
4: 0.0023286, 0.0040240, 0.0023286, 0.0040240, -0.0016955, 0.0016955
5: 0.0051698, 0.0071672, 0.0051698, 0.0071672, -0.0019974, 0.0019974
6: -0.0015836, -0.0006646, -0.0015836, -0.0006646, -0.0009190, 0.0009190
7: -0.0087729, -0.0073347, -0.0087729, -0.0073347, -0.0014382, 0.0014382
8: 0.0030267, 0.0075728, 0.0030267, 0.0075728, -0.0039055, 0.0039055
9: -0.0047209, -0.0021125, -0.0047209, -0.0021125, -0.0026083, 0.0026083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011516, upper bound: 0.0011152
time: 1.02 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011509, upper bound: 0.0011194
time: 1.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.35 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 1, lower bound: -0.0010854, upper bound: 0.0010758
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 1, lower bound: -0.0010973, upper bound: 0.0010987
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 1, lower bound: -0.0010854, upper bound: 0.0011332
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 1, lower bound: -0.0010973, upper bound: 0.0011573
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 1, lower bound: -0.0011332, upper bound: 0.0010854
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 1, lower bound: -0.0011573, upper bound: 0.0010973
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 1, lower bound: -0.0011516, upper bound: 0.0011152
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 1, lower bound: -0.0011509, upper bound: 0.0011194

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006265, 0.0013554, 0.0006029, 0.0013435, -0.0007078, 0.0007434
1: 0.9930440, 0.9948819, 0.9930741, 0.9949416, -0.0017841, 0.0016950
2: -0.0068180, -0.0045638, -0.0067656, -0.0045078, -0.0023102, 0.0022018
3: 0.0034666, 0.0042289, 0.0034419, 0.0042164, -0.0006732, 0.0007102
4: 0.0023554, 0.0038056, 0.0023279, 0.0037642, -0.0014088, 0.0014776
5: 0.0052264, 0.0070128, 0.0051684, 0.0069835, -0.0017571, 0.0018443
6: -0.0014877, -0.0006846, -0.0014695, -0.0006641, -0.0008236, 0.0007849
7: -0.0086617, -0.0073754, -0.0086406, -0.0073337, -0.0013280, 0.0012652
8: 0.0031527, 0.0072096, 0.0030236, 0.0071408, -0.0034784, 0.0036697
9: -0.0045192, -0.0021865, -0.0044810, -0.0021108, -0.0024084, 0.0022945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010587, upper bound: 0.0010503
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010603, upper bound: 0.0010515
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006098, 0.0013434, 0.0005984, 0.0013435, -0.0007244, 0.0007359
1: 0.9930743, 0.9949241, 0.9930739, 0.9949527, -0.0017661, 0.0017351
2: -0.0067654, -0.0045243, -0.0067658, -0.0044973, -0.0022681, 0.0022415
3: 0.0034492, 0.0042164, 0.0034372, 0.0042165, -0.0006890, 0.0007029
4: 0.0023360, 0.0037640, 0.0023228, 0.0037643, -0.0014283, 0.0014412
5: 0.0051856, 0.0069834, 0.0051576, 0.0069836, -0.0017980, 0.0018258
6: -0.0014694, -0.0006702, -0.0014696, -0.0006603, -0.0008091, 0.0007994
7: -0.0086405, -0.0073460, -0.0086407, -0.0073259, -0.0013146, 0.0012947
8: 0.0030618, 0.0071405, 0.0029996, 0.0071411, -0.0035320, 0.0036542
9: -0.0044808, -0.0021331, -0.0044811, -0.0020966, -0.0023842, 0.0023479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010728
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010734, upper bound: 0.0010734
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006265, 0.0013554, 0.0006079, 0.0014184, -0.0007829, 0.0007386
1: 0.9930440, 0.9948819, 0.9928851, 0.9949291, -0.0017746, 0.0018861
2: -0.0068180, -0.0045638, -0.0070942, -0.0045197, -0.0022983, 0.0025304
3: 0.0034666, 0.0042289, 0.0034471, 0.0042948, -0.0007530, 0.0007067
4: 0.0023554, 0.0038056, 0.0023338, 0.0040239, -0.0016685, 0.0014718
5: 0.0052264, 0.0070128, 0.0051808, 0.0071671, -0.0019407, 0.0018320
6: -0.0014877, -0.0006846, -0.0015835, -0.0006685, -0.0008192, 0.0008989
7: -0.0086617, -0.0073754, -0.0087728, -0.0073425, -0.0013191, 0.0013974
8: 0.0031527, 0.0072096, 0.0030511, 0.0075725, -0.0038928, 0.0036506
9: -0.0045192, -0.0021865, -0.0047207, -0.0021269, -0.0023923, 0.0025342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0011046
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010603, upper bound: 0.0011088
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006098, 0.0013434, 0.0006034, 0.0014184, -0.0007995, 0.0007311
1: 0.9930743, 0.9949241, 0.9928850, 0.9949400, -0.0017567, 0.0019259
2: -0.0067654, -0.0045243, -0.0070944, -0.0045091, -0.0022563, 0.0025701
3: 0.0034492, 0.0042164, 0.0034424, 0.0042948, -0.0007688, 0.0006995
4: 0.0023360, 0.0037640, 0.0023286, 0.0040240, -0.0016880, 0.0014354
5: 0.0051856, 0.0069834, 0.0051698, 0.0071672, -0.0019816, 0.0018136
6: -0.0014694, -0.0006702, -0.0015836, -0.0006646, -0.0008048, 0.0009134
7: -0.0086405, -0.0073460, -0.0087729, -0.0073347, -0.0013059, 0.0014269
8: 0.0030618, 0.0071405, 0.0030267, 0.0075728, -0.0039575, 0.0036333
9: -0.0044808, -0.0021331, -0.0047209, -0.0021125, -0.0023683, 0.0025877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010719, upper bound: 0.0011329
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010725, upper bound: 0.0011316
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0006079, 0.0014184, 0.0006265, 0.0013554, -0.0007386, 0.0007829
1: 0.9928851, 0.9949291, 0.9930440, 0.9948819, -0.0018861, 0.0017746
2: -0.0070942, -0.0045197, -0.0068180, -0.0045638, -0.0025304, 0.0022983
3: 0.0034471, 0.0042948, 0.0034666, 0.0042289, -0.0007067, 0.0007530
4: 0.0023338, 0.0040239, 0.0023554, 0.0038056, -0.0014718, 0.0016685
5: 0.0051808, 0.0071671, 0.0052264, 0.0070128, -0.0018320, 0.0019407
6: -0.0015835, -0.0006685, -0.0014877, -0.0006846, -0.0008989, 0.0008192
7: -0.0087728, -0.0073425, -0.0086617, -0.0073754, -0.0013974, 0.0013191
8: 0.0030511, 0.0075725, 0.0031527, 0.0072096, -0.0036506, 0.0038928
9: -0.0047207, -0.0021269, -0.0045192, -0.0021865, -0.0025342, 0.0023923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011046, upper bound: 0.0010589
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011088, upper bound: 0.0010603
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0006034, 0.0014184, 0.0006098, 0.0013434, -0.0007311, 0.0007995
1: 0.9928850, 0.9949400, 0.9930743, 0.9949241, -0.0019259, 0.0017567
2: -0.0070944, -0.0045091, -0.0067654, -0.0045243, -0.0025701, 0.0022563
3: 0.0034424, 0.0042948, 0.0034492, 0.0042164, -0.0006995, 0.0007688
4: 0.0023286, 0.0040240, 0.0023360, 0.0037640, -0.0014354, 0.0016880
5: 0.0051698, 0.0071672, 0.0051856, 0.0069834, -0.0018136, 0.0019816
6: -0.0015836, -0.0006646, -0.0014694, -0.0006702, -0.0009134, 0.0008048
7: -0.0087729, -0.0073347, -0.0086405, -0.0073460, -0.0014269, 0.0013059
8: 0.0030267, 0.0075728, 0.0030618, 0.0071405, -0.0036333, 0.0039575
9: -0.0047209, -0.0021125, -0.0044808, -0.0021331, -0.0025877, 0.0023683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011329, upper bound: 0.0010719
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011316, upper bound: 0.0010725
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0006064, 0.0014184, 0.0006260, 0.0014258, -0.0008100, 0.0007830
1: 0.9928851, 0.9949327, 0.9928665, 0.9948833, -0.0018813, 0.0019485
2: -0.0070942, -0.0045161, -0.0071269, -0.0045625, -0.0025317, 0.0026108
3: 0.0034455, 0.0042948, 0.0034660, 0.0043026, -0.0007766, 0.0007486
4: 0.0023320, 0.0040239, 0.0023548, 0.0040497, -0.0017177, 0.0016691
5: 0.0051770, 0.0071671, 0.0052251, 0.0071854, -0.0020083, 0.0019420
6: -0.0015835, -0.0006672, -0.0015949, -0.0006841, -0.0008994, 0.0009277
7: -0.0087728, -0.0073399, -0.0087860, -0.0073745, -0.0013983, 0.0014461
8: 0.0030428, 0.0075725, 0.0031498, 0.0076155, -0.0038541, 0.0037279
9: -0.0047207, -0.0021220, -0.0047446, -0.0021847, -0.0025359, 0.0026226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011232, upper bound: 0.0010794
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011393, upper bound: 0.0010966
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0006034, 0.0014184, 0.0006202, 0.0014182, -0.0008054, 0.0007887
1: 0.9928850, 0.9949400, 0.9928854, 0.9948976, -0.0018978, 0.0019381
2: -0.0070944, -0.0045091, -0.0070936, -0.0045490, -0.0025454, 0.0025845
3: 0.0034424, 0.0042948, 0.0034601, 0.0042946, -0.0007725, 0.0007563
4: 0.0023286, 0.0040240, 0.0023481, 0.0040234, -0.0016948, 0.0016759
5: 0.0051698, 0.0071672, 0.0052111, 0.0071668, -0.0019969, 0.0019561
6: -0.0015836, -0.0006646, -0.0015833, -0.0006792, -0.0009044, 0.0009187
7: -0.0087729, -0.0073347, -0.0087726, -0.0073644, -0.0014085, 0.0014379
8: 0.0030267, 0.0075728, 0.0031186, 0.0075717, -0.0038472, 0.0038043
9: -0.0047209, -0.0021125, -0.0047203, -0.0021664, -0.0025544, 0.0026077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011284, upper bound: 0.0010847
time: 1.02 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011409, upper bound: 0.0011005
time: 1.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0010587, upper bound: 0.0010503
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0010603, upper bound: 0.0010515
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010728
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0010734, upper bound: 0.0010734
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0011046
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0010603, upper bound: 0.0011088
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0010719, upper bound: 0.0011329
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0010725, upper bound: 0.0011316
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0011046, upper bound: 0.0010589
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0011088, upper bound: 0.0010603
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0011329, upper bound: 0.0010719
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0011316, upper bound: 0.0010725
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0011232, upper bound: 0.0010794
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0011393, upper bound: 0.0010966
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0011284, upper bound: 0.0010847
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 1, lower bound: -0.0011409, upper bound: 0.0011005

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006297, 0.0013554, 0.0006264, 0.0013501, -0.0007111, 0.0007200
1: 0.9930440, 0.9948739, 0.9930574, 0.9948824, -0.0017256, 0.0017022
2: -0.0068180, -0.0045714, -0.0067947, -0.0045634, -0.0022546, 0.0022233
3: 0.0034699, 0.0042289, 0.0034664, 0.0042234, -0.0006757, 0.0006855
4: 0.0023591, 0.0038056, 0.0023552, 0.0037872, -0.0014280, 0.0014503
5: 0.0052343, 0.0070128, 0.0052260, 0.0069997, -0.0017655, 0.0017867
6: -0.0014877, -0.0006874, -0.0014796, -0.0006845, -0.0008032, 0.0007922
7: -0.0086617, -0.0073811, -0.0086523, -0.0073751, -0.0012865, 0.0012712
8: 0.0031702, 0.0072096, 0.0031518, 0.0071790, -0.0033919, 0.0034861
9: -0.0045191, -0.0021967, -0.0045022, -0.0021860, -0.0023332, 0.0023054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010254, upper bound: 0.0010121
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010410, upper bound: 0.0010330
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006265, 0.0013554, 0.0006190, 0.0013434, -0.0007078, 0.0007272
1: 0.9930440, 0.9948819, 0.9930742, 0.9949008, -0.0017430, 0.0016949
2: -0.0068180, -0.0045638, -0.0067654, -0.0045460, -0.0022719, 0.0022016
3: 0.0034666, 0.0042289, 0.0034588, 0.0042164, -0.0006731, 0.0006928
4: 0.0023554, 0.0038056, 0.0023467, 0.0037640, -0.0014086, 0.0014589
5: 0.0052264, 0.0070128, 0.0052081, 0.0069834, -0.0017570, 0.0018047
6: -0.0014877, -0.0006846, -0.0014694, -0.0006781, -0.0008096, 0.0007848
7: -0.0086617, -0.0073754, -0.0086405, -0.0073622, -0.0012995, 0.0012651
8: 0.0031527, 0.0072096, 0.0031118, 0.0071405, -0.0034122, 0.0035157
9: -0.0045192, -0.0021865, -0.0044808, -0.0021625, -0.0023567, 0.0022943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0010477
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0010515
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006129, 0.0013434, 0.0006218, 0.0013501, -0.0007278, 0.0007126
1: 0.9930743, 0.9949159, 0.9930573, 0.9948936, -0.0017077, 0.0017422
2: -0.0067654, -0.0045317, -0.0067949, -0.0045527, -0.0022126, 0.0022632
3: 0.0034524, 0.0042164, 0.0034617, 0.0042234, -0.0006916, 0.0006783
4: 0.0023396, 0.0037640, 0.0023500, 0.0037873, -0.0014477, 0.0014140
5: 0.0051932, 0.0069834, 0.0052150, 0.0069999, -0.0018067, 0.0017684
6: -0.0014694, -0.0006729, -0.0014796, -0.0006806, -0.0007889, 0.0008068
7: -0.0086405, -0.0073515, -0.0086524, -0.0073672, -0.0012733, 0.0013009
8: 0.0030787, 0.0071405, 0.0031272, 0.0071792, -0.0034446, 0.0034708
9: -0.0044808, -0.0021431, -0.0045023, -0.0021715, -0.0023093, 0.0023592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010589
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010728
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006098, 0.0013434, 0.0006146, 0.0013435, -0.0007244, 0.0007197
1: 0.9930743, 0.9949241, 0.9930741, 0.9949119, -0.0017260, 0.0017350
2: -0.0067654, -0.0045243, -0.0067656, -0.0045356, -0.0022298, 0.0022412
3: 0.0034492, 0.0042164, 0.0034542, 0.0042164, -0.0006890, 0.0006864
4: 0.0023360, 0.0037640, 0.0023416, 0.0037641, -0.0014281, 0.0014224
5: 0.0051856, 0.0069834, 0.0051973, 0.0069835, -0.0017979, 0.0017861
6: -0.0014694, -0.0006702, -0.0014695, -0.0006743, -0.0007951, 0.0007993
7: -0.0086405, -0.0073460, -0.0086406, -0.0073544, -0.0012861, 0.0012946
8: 0.0030618, 0.0071405, 0.0030878, 0.0071407, -0.0034638, 0.0035007
9: -0.0044808, -0.0021331, -0.0044809, -0.0021484, -0.0023324, 0.0023478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010515, upper bound: 0.0010603
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010515, upper bound: 0.0010734
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0006507, 0.0013610, 0.0006108, 0.0014183, -0.0007587, 0.0007410
1: 0.9930300, 0.9948210, 0.9928852, 0.9949214, -0.0017797, 0.0018254
2: -0.0068423, -0.0046210, -0.0070940, -0.0045267, -0.0023156, 0.0024729
3: 0.0034919, 0.0042347, 0.0034502, 0.0042947, -0.0007276, 0.0007085
4: 0.0023835, 0.0038248, 0.0023372, 0.0040237, -0.0016402, 0.0014876
5: 0.0052857, 0.0070264, 0.0051880, 0.0071670, -0.0018813, 0.0018383
6: -0.0014961, -0.0007056, -0.0015835, -0.0006710, -0.0008251, 0.0008779
7: -0.0086715, -0.0074181, -0.0087727, -0.0073478, -0.0013237, 0.0013546
8: 0.0032847, 0.0072416, 0.0030673, 0.0075723, -0.0037056, 0.0035691
9: -0.0045369, -0.0022639, -0.0047205, -0.0021364, -0.0024006, 0.0024567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010225, upper bound: 0.0010619
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010414, upper bound: 0.0010874
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0006425, 0.0013554, 0.0006079, 0.0014184, -0.0007669, 0.0007385
1: 0.9930440, 0.9948417, 0.9928851, 0.9949291, -0.0017745, 0.0018470
2: -0.0068178, -0.0046017, -0.0070942, -0.0045197, -0.0022981, 0.0024924
3: 0.0034834, 0.0042289, 0.0034471, 0.0042948, -0.0007371, 0.0007067
4: 0.0023740, 0.0038054, 0.0023338, 0.0040239, -0.0016499, 0.0014717
5: 0.0052657, 0.0070127, 0.0051808, 0.0071671, -0.0019014, 0.0018319
6: -0.0014876, -0.0006985, -0.0015835, -0.0006685, -0.0008191, 0.0008851
7: -0.0086616, -0.0074037, -0.0087728, -0.0073425, -0.0013191, 0.0013691
8: 0.0032402, 0.0072094, 0.0030511, 0.0075725, -0.0037930, 0.0035825
9: -0.0045190, -0.0022378, -0.0047207, -0.0021269, -0.0023922, 0.0024829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010584, upper bound: 0.0011020
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010584, upper bound: 0.0011088
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0006333, 0.0013500, 0.0006064, 0.0014184, -0.0007760, 0.0007347
1: 0.9930574, 0.9948649, 0.9928851, 0.9949327, -0.0017646, 0.0018671
2: -0.0067945, -0.0045798, -0.0070942, -0.0045161, -0.0022784, 0.0025144
3: 0.0034737, 0.0042233, 0.0034455, 0.0042948, -0.0007441, 0.0007022
4: 0.0023632, 0.0037870, 0.0023320, 0.0040239, -0.0016606, 0.0014550
5: 0.0052430, 0.0069996, 0.0051770, 0.0071671, -0.0019241, 0.0018226
6: -0.0014795, -0.0006905, -0.0015835, -0.0006672, -0.0008123, 0.0008931
7: -0.0086522, -0.0073874, -0.0087728, -0.0073399, -0.0013123, 0.0013855
8: 0.0031896, 0.0071787, 0.0030428, 0.0075725, -0.0037713, 0.0035527
9: -0.0045020, -0.0022081, -0.0047207, -0.0021220, -0.0023800, 0.0025126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010504, upper bound: 0.0011184
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010504, upper bound: 0.0011329
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006034, 0.0014184, -0.0007832, 0.0007311
1: 0.9930742, 0.9948826, 0.9928850, 0.9949400, -0.0017565, 0.0018862
2: -0.0067652, -0.0045631, -0.0070944, -0.0045091, -0.0022561, 0.0025313
3: 0.0034663, 0.0042163, 0.0034424, 0.0042948, -0.0007529, 0.0006994
4: 0.0023551, 0.0037638, 0.0023286, 0.0040240, -0.0016690, 0.0014353
5: 0.0052257, 0.0069833, 0.0051698, 0.0071672, -0.0019415, 0.0018134
6: -0.0014693, -0.0006844, -0.0015836, -0.0006646, -0.0008047, 0.0008993
7: -0.0086404, -0.0073749, -0.0087729, -0.0073347, -0.0013058, 0.0013980
8: 0.0031512, 0.0071402, 0.0030267, 0.0075728, -0.0038527, 0.0035671
9: -0.0044806, -0.0021856, -0.0047209, -0.0021125, -0.0023681, 0.0025353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010711, upper bound: 0.0011280
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010711, upper bound: 0.0011316
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0006108, 0.0014183, 0.0006507, 0.0013610, -0.0007410, 0.0007587
1: 0.9928852, 0.9949214, 0.9930300, 0.9948210, -0.0018254, 0.0017797
2: -0.0070940, -0.0045267, -0.0068423, -0.0046210, -0.0024729, 0.0023156
3: 0.0034502, 0.0042947, 0.0034919, 0.0042347, -0.0007085, 0.0007276
4: 0.0023372, 0.0040237, 0.0023835, 0.0038248, -0.0014876, 0.0016402
5: 0.0051880, 0.0071670, 0.0052857, 0.0070264, -0.0018383, 0.0018813
6: -0.0015835, -0.0006710, -0.0014961, -0.0007056, -0.0008779, 0.0008251
7: -0.0087727, -0.0073478, -0.0086715, -0.0074181, -0.0013546, 0.0013237
8: 0.0030673, 0.0075723, 0.0032847, 0.0072416, -0.0035691, 0.0037056
9: -0.0047205, -0.0021364, -0.0045369, -0.0022639, -0.0024567, 0.0024006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010619, upper bound: 0.0010225
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010874, upper bound: 0.0010414
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0006079, 0.0014184, 0.0006425, 0.0013554, -0.0007385, 0.0007669
1: 0.9928851, 0.9949291, 0.9930440, 0.9948417, -0.0018470, 0.0017745
2: -0.0070942, -0.0045197, -0.0068178, -0.0046017, -0.0024924, 0.0022981
3: 0.0034471, 0.0042948, 0.0034834, 0.0042289, -0.0007067, 0.0007371
4: 0.0023338, 0.0040239, 0.0023740, 0.0038054, -0.0014717, 0.0016499
5: 0.0051808, 0.0071671, 0.0052657, 0.0070127, -0.0018319, 0.0019014
6: -0.0015835, -0.0006685, -0.0014876, -0.0006985, -0.0008851, 0.0008191
7: -0.0087728, -0.0073425, -0.0086616, -0.0074037, -0.0013691, 0.0013191
8: 0.0030511, 0.0075725, 0.0032402, 0.0072094, -0.0035825, 0.0037930
9: -0.0047207, -0.0021269, -0.0045190, -0.0022378, -0.0024829, 0.0023922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011020, upper bound: 0.0010584
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011020, upper bound: 0.0010603
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0006064, 0.0014184, 0.0006333, 0.0013500, -0.0007347, 0.0007760
1: 0.9928851, 0.9949327, 0.9930574, 0.9948649, -0.0018671, 0.0017646
2: -0.0070942, -0.0045161, -0.0067945, -0.0045798, -0.0025144, 0.0022784
3: 0.0034455, 0.0042948, 0.0034737, 0.0042233, -0.0007022, 0.0007441
4: 0.0023320, 0.0040239, 0.0023632, 0.0037870, -0.0014550, 0.0016606
5: 0.0051770, 0.0071671, 0.0052430, 0.0069996, -0.0018226, 0.0019241
6: -0.0015835, -0.0006672, -0.0014795, -0.0006905, -0.0008931, 0.0008123
7: -0.0087728, -0.0073399, -0.0086522, -0.0073874, -0.0013855, 0.0013123
8: 0.0030428, 0.0075725, 0.0031896, 0.0071787, -0.0035527, 0.0037713
9: -0.0047207, -0.0021220, -0.0045020, -0.0022081, -0.0025126, 0.0023800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011041, upper bound: 0.0010503
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011041, upper bound: 0.0010719
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0006034, 0.0014184, 0.0006262, 0.0013434, -0.0007311, 0.0007832
1: 0.9928850, 0.9949400, 0.9930742, 0.9948826, -0.0018862, 0.0017565
2: -0.0070944, -0.0045091, -0.0067652, -0.0045631, -0.0025313, 0.0022561
3: 0.0034424, 0.0042948, 0.0034663, 0.0042163, -0.0006994, 0.0007529
4: 0.0023286, 0.0040240, 0.0023551, 0.0037638, -0.0014353, 0.0016690
5: 0.0051698, 0.0071672, 0.0052257, 0.0069833, -0.0018134, 0.0019415
6: -0.0015836, -0.0006646, -0.0014693, -0.0006844, -0.0008993, 0.0008047
7: -0.0087729, -0.0073347, -0.0086404, -0.0073749, -0.0013980, 0.0013058
8: 0.0030267, 0.0075728, 0.0031512, 0.0071402, -0.0035671, 0.0038528
9: -0.0047209, -0.0021125, -0.0044806, -0.0021856, -0.0025353, 0.0023681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011280, upper bound: 0.0010711
time: 1.00 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011280, upper bound: 0.0010725
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006349, 0.0014293, 0.0006305, 0.0014258, -0.0007812, 0.0007893
1: 0.9928576, 0.9948608, 0.9928665, 0.9948716, -0.0018963, 0.0018752
2: -0.0071421, -0.0045837, -0.0071267, -0.0045733, -0.0025688, 0.0025430
3: 0.0034754, 0.0043062, 0.0034708, 0.0043025, -0.0007462, 0.0007546
4: 0.0023652, 0.0040617, 0.0023601, 0.0040496, -0.0016844, 0.0017017
5: 0.0052471, 0.0071939, 0.0052363, 0.0071853, -0.0019382, 0.0019575
6: -0.0016002, -0.0006919, -0.0015948, -0.0006881, -0.0009121, 0.0009029
7: -0.0087921, -0.0073903, -0.0087859, -0.0073826, -0.0014095, 0.0013956
8: 0.0031986, 0.0076355, 0.0031747, 0.0076152, -0.0036797, 0.0037501
9: -0.0047557, -0.0022134, -0.0047444, -0.0021994, -0.0025563, 0.0025310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010896, upper bound: 0.0010477
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011059, upper bound: 0.0010620
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006177, 0.0014183, 0.0006260, 0.0014258, -0.0007984, 0.0007829
1: 0.9928854, 0.9949040, 0.9928665, 0.9948833, -0.0018810, 0.0019168
2: -0.0070937, -0.0045430, -0.0071269, -0.0045625, -0.0025312, 0.0025839
3: 0.0034574, 0.0042947, 0.0034660, 0.0043026, -0.0007625, 0.0007485
4: 0.0023452, 0.0040235, 0.0023548, 0.0040497, -0.0017045, 0.0016687
5: 0.0052049, 0.0071669, 0.0052251, 0.0071854, -0.0019805, 0.0019417
6: -0.0015834, -0.0006770, -0.0015949, -0.0006841, -0.0008993, 0.0009179
7: -0.0087726, -0.0073599, -0.0087860, -0.0073745, -0.0013981, 0.0014260
8: 0.0031048, 0.0075719, 0.0031498, 0.0076155, -0.0037390, 0.0037264
9: -0.0047204, -0.0021584, -0.0047446, -0.0021847, -0.0025356, 0.0025862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011096, upper bound: 0.0010857
time: 0.94 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011096, upper bound: 0.0010966
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006319, 0.0014294, 0.0006247, 0.0014182, -0.0007767, 0.0007951
1: 0.9928575, 0.9948682, 0.9928857, 0.9948865, -0.0019133, 0.0018649
2: -0.0071423, -0.0045765, -0.0070934, -0.0045595, -0.0025828, 0.0025168
3: 0.0034722, 0.0043063, 0.0034647, 0.0042946, -0.0007421, 0.0007623
4: 0.0023617, 0.0040619, 0.0023533, 0.0040232, -0.0016616, 0.0017086
5: 0.0052396, 0.0071940, 0.0052220, 0.0071666, -0.0019270, 0.0019720
6: -0.0016002, -0.0006893, -0.0015833, -0.0006830, -0.0009172, 0.0008940
7: -0.0087922, -0.0073849, -0.0087725, -0.0073722, -0.0014199, 0.0013875
8: 0.0031821, 0.0076357, 0.0031428, 0.0075714, -0.0036721, 0.0038201
9: -0.0047558, -0.0022037, -0.0047201, -0.0021807, -0.0025751, 0.0025164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011203, upper bound: 0.0010747
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011203, upper bound: 0.0010847
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006148, 0.0014183, 0.0006202, 0.0014182, -0.0007938, 0.0007886
1: 0.9928852, 0.9949114, 0.9928854, 0.9948976, -0.0018975, 0.0019066
2: -0.0070940, -0.0045361, -0.0070936, -0.0045490, -0.0025450, 0.0025575
3: 0.0034544, 0.0042947, 0.0034601, 0.0042946, -0.0007585, 0.0007562
4: 0.0023418, 0.0040237, 0.0023481, 0.0040234, -0.0016816, 0.0016756
5: 0.0051978, 0.0071670, 0.0052111, 0.0071668, -0.0019690, 0.0019559
6: -0.0015835, -0.0006745, -0.0015833, -0.0006792, -0.0009043, 0.0009089
7: -0.0087727, -0.0073548, -0.0087726, -0.0073644, -0.0014083, 0.0014178
8: 0.0030889, 0.0075722, 0.0031186, 0.0075717, -0.0037343, 0.0038029
9: -0.0047205, -0.0021490, -0.0047203, -0.0021664, -0.0025541, 0.0025712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011349, upper bound: 0.0010953
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011349, upper bound: 0.0011005
time: 1.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.67 seconds
IS_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010254, upper bound: 0.0010121
IS_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010410, upper bound: 0.0010330
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0010477
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010589, upper bound: 0.0010515
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010589
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010728
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010515, upper bound: 0.0010603
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010515, upper bound: 0.0010734
IS_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010225, upper bound: 0.0010619
IS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010414, upper bound: 0.0010874
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010584, upper bound: 0.0011020
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010584, upper bound: 0.0011088
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010504, upper bound: 0.0011184
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010504, upper bound: 0.0011329
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010711, upper bound: 0.0011280
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010711, upper bound: 0.0011316
IS_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010619, upper bound: 0.0010225
IS_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010874, upper bound: 0.0010414
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011020, upper bound: 0.0010584
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011020, upper bound: 0.0010603
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011041, upper bound: 0.0010503
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011041, upper bound: 0.0010719
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011280, upper bound: 0.0010711
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011280, upper bound: 0.0010725
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0010896, upper bound: 0.0010477
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011059, upper bound: 0.0010620
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011096, upper bound: 0.0010857
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011096, upper bound: 0.0010966
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011203, upper bound: 0.0010747
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011203, upper bound: 0.0010847
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011349, upper bound: 0.0010953
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 1, lower bound: -0.0011349, upper bound: 0.0011005

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006507, 0.0013610, 0.0006190, 0.0013434, -0.0006837, 0.0007326
1: 0.9930300, 0.9948210, 0.9930742, 0.9949008, -0.0017574, 0.0016344
2: -0.0068423, -0.0046210, -0.0067654, -0.0045460, -0.0022963, 0.0021443
3: 0.0034919, 0.0042347, 0.0034588, 0.0042164, -0.0006478, 0.0006993
4: 0.0023835, 0.0038248, 0.0023467, 0.0037640, -0.0013805, 0.0014781
5: 0.0052857, 0.0070264, 0.0052081, 0.0069834, -0.0016977, 0.0018183
6: -0.0014961, -0.0007056, -0.0014694, -0.0006781, -0.0008180, 0.0007639
7: -0.0086715, -0.0074181, -0.0086405, -0.0073622, -0.0013093, 0.0012224
8: 0.0032847, 0.0072416, 0.0031118, 0.0071405, -0.0032349, 0.0035204
9: -0.0045369, -0.0022639, -0.0044808, -0.0021625, -0.0023745, 0.0022169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010234, upper bound: 0.0010100
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010406, upper bound: 0.0010299
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006425, 0.0013554, 0.0006190, 0.0013434, -0.0006917, 0.0007271
1: 0.9930440, 0.9948417, 0.9930742, 0.9949008, -0.0017428, 0.0016549
2: -0.0068178, -0.0046017, -0.0067654, -0.0045460, -0.0022717, 0.0021636
3: 0.0034834, 0.0042289, 0.0034588, 0.0042164, -0.0006565, 0.0006927
4: 0.0023740, 0.0038054, 0.0023467, 0.0037640, -0.0013900, 0.0014587
5: 0.0052657, 0.0070127, 0.0052081, 0.0069834, -0.0017177, 0.0018046
6: -0.0014876, -0.0006985, -0.0014694, -0.0006781, -0.0008095, 0.0007709
7: -0.0086616, -0.0074037, -0.0086405, -0.0073622, -0.0012994, 0.0012368
8: 0.0032402, 0.0072094, 0.0031118, 0.0071405, -0.0032768, 0.0034680
9: -0.0045190, -0.0022378, -0.0044808, -0.0021625, -0.0023565, 0.0022430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010234, upper bound: 0.0010121
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010406, upper bound: 0.0010346
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0006129, 0.0013434, 0.0006507, 0.0013610, -0.0007388, 0.0006837
1: 0.9930743, 0.9949159, 0.9930300, 0.9948210, -0.0016344, 0.0017723
2: -0.0067654, -0.0045317, -0.0068423, -0.0046210, -0.0021443, 0.0023107
3: 0.0034524, 0.0042164, 0.0034919, 0.0042347, -0.0007049, 0.0006478
4: 0.0023396, 0.0037640, 0.0023835, 0.0038248, -0.0014852, 0.0013805
5: 0.0051932, 0.0069834, 0.0052857, 0.0070264, -0.0018332, 0.0016977
6: -0.0014694, -0.0006729, -0.0014961, -0.0007056, -0.0007639, 0.0008233
7: -0.0086405, -0.0073515, -0.0086715, -0.0074181, -0.0012224, 0.0013200
8: 0.0030787, 0.0071405, 0.0032847, 0.0072416, -0.0035745, 0.0032930
9: -0.0044808, -0.0021431, -0.0045369, -0.0022639, -0.0022169, 0.0023939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010234
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010414
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0006129, 0.0013434, 0.0006333, 0.0013500, -0.0007277, 0.0007009
1: 0.9930743, 0.9949159, 0.9930574, 0.9948649, -0.0016762, 0.0017420
2: -0.0067654, -0.0045317, -0.0067945, -0.0045798, -0.0021856, 0.0022628
3: 0.0034524, 0.0042164, 0.0034737, 0.0042233, -0.0006915, 0.0006642
4: 0.0023396, 0.0037640, 0.0023632, 0.0037870, -0.0014473, 0.0014008
5: 0.0051932, 0.0069834, 0.0052430, 0.0069996, -0.0018064, 0.0017404
6: -0.0014694, -0.0006729, -0.0014795, -0.0006905, -0.0007790, 0.0008066
7: -0.0086405, -0.0073515, -0.0086522, -0.0073874, -0.0012532, 0.0013007
8: 0.0030787, 0.0071405, 0.0031896, 0.0071787, -0.0034432, 0.0033471
9: -0.0044808, -0.0021431, -0.0045020, -0.0022081, -0.0022727, 0.0023589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010377
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010561
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0006098, 0.0013434, 0.0006425, 0.0013554, -0.0007365, 0.0006918
1: 0.9930743, 0.9949241, 0.9930440, 0.9948417, -0.0016549, 0.0017674
2: -0.0067654, -0.0045243, -0.0068178, -0.0046017, -0.0021637, 0.0022934
3: 0.0034492, 0.0042164, 0.0034834, 0.0042289, -0.0007033, 0.0006566
4: 0.0023360, 0.0037640, 0.0023740, 0.0038054, -0.0014694, 0.0013900
5: 0.0051856, 0.0069834, 0.0052657, 0.0070127, -0.0018271, 0.0017177
6: -0.0014694, -0.0006702, -0.0014876, -0.0006985, -0.0007709, 0.0008174
7: -0.0086405, -0.0073460, -0.0086616, -0.0074037, -0.0012368, 0.0013156
8: 0.0030618, 0.0071405, 0.0032402, 0.0072094, -0.0035925, 0.0033225
9: -0.0044808, -0.0021331, -0.0045190, -0.0022378, -0.0022430, 0.0023859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010585
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010603
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0006098, 0.0013434, 0.0006262, 0.0013434, -0.0007243, 0.0007080
1: 0.9930743, 0.9949241, 0.9930742, 0.9948826, -0.0016944, 0.0017348
2: -0.0067654, -0.0045243, -0.0067652, -0.0045631, -0.0022023, 0.0022408
3: 0.0034492, 0.0042164, 0.0034663, 0.0042163, -0.0006889, 0.0006723
4: 0.0023360, 0.0037640, 0.0023551, 0.0037638, -0.0014278, 0.0014090
5: 0.0051856, 0.0069834, 0.0052257, 0.0069833, -0.0017977, 0.0017577
6: -0.0014694, -0.0006702, -0.0014693, -0.0006844, -0.0007851, 0.0007992
7: -0.0086405, -0.0073460, -0.0086404, -0.0073749, -0.0012656, 0.0012944
8: 0.0030618, 0.0071405, 0.0031512, 0.0071402, -0.0034624, 0.0033749
9: -0.0044808, -0.0021331, -0.0044806, -0.0021856, -0.0022952, 0.0023475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010725
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010734
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0006730, 0.0013796, 0.0006150, 0.0014183, -0.0007362, 0.0007554
1: 0.9929829, 0.9947646, 0.9928854, 0.9949108, -0.0018145, 0.0017670
2: -0.0069241, -0.0046738, -0.0070938, -0.0045365, -0.0023876, 0.0024200
3: 0.0035152, 0.0042542, 0.0034546, 0.0042947, -0.0007029, 0.0007225
4: 0.0024094, 0.0038894, 0.0023420, 0.0040236, -0.0016142, 0.0015474
5: 0.0053403, 0.0070721, 0.0051982, 0.0071669, -0.0018265, 0.0018738
6: -0.0015245, -0.0007249, -0.0015834, -0.0006746, -0.0008499, 0.0008585
7: -0.0087044, -0.0074575, -0.0087726, -0.0073551, -0.0013493, 0.0013152
8: 0.0034063, 0.0073490, 0.0030899, 0.0075720, -0.0035431, 0.0036036
9: -0.0045966, -0.0023352, -0.0047204, -0.0021497, -0.0024469, 0.0023852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010612
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010619
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0006615, 0.0013609, 0.0006108, 0.0014183, -0.0007479, 0.0007409
1: 0.9930301, 0.9947939, 0.9928852, 0.9949214, -0.0017796, 0.0017981
2: -0.0068421, -0.0046465, -0.0070940, -0.0045267, -0.0023154, 0.0024474
3: 0.0035032, 0.0042347, 0.0034502, 0.0042947, -0.0007160, 0.0007084
4: 0.0023960, 0.0038246, 0.0023372, 0.0040237, -0.0016277, 0.0014874
5: 0.0053121, 0.0070262, 0.0051880, 0.0071670, -0.0018549, 0.0018382
6: -0.0014960, -0.0007149, -0.0015835, -0.0006710, -0.0008250, 0.0008686
7: -0.0086714, -0.0074371, -0.0087727, -0.0073478, -0.0013236, 0.0013356
8: 0.0033434, 0.0072413, 0.0030673, 0.0075723, -0.0035708, 0.0035678
9: -0.0045368, -0.0022983, -0.0047205, -0.0021364, -0.0024004, 0.0024222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010405, upper bound: 0.0010851
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010405, upper bound: 0.0010874
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006425, 0.0013554, 0.0006305, 0.0014258, -0.0007742, 0.0007159
1: 0.9930440, 0.9948417, 0.9928665, 0.9948716, -0.0017172, 0.0018650
2: -0.0068178, -0.0046017, -0.0071267, -0.0045733, -0.0022444, 0.0025249
3: 0.0034834, 0.0042289, 0.0034708, 0.0043025, -0.0007444, 0.0006827
4: 0.0023740, 0.0038054, 0.0023601, 0.0040496, -0.0016756, 0.0014453
5: 0.0052657, 0.0070127, 0.0052363, 0.0071853, -0.0019195, 0.0017763
6: -0.0014876, -0.0006985, -0.0015948, -0.0006881, -0.0007995, 0.0008963
7: -0.0086616, -0.0074037, -0.0087859, -0.0073826, -0.0012791, 0.0013822
8: 0.0032402, 0.0072094, 0.0031747, 0.0076152, -0.0037890, 0.0034244
9: -0.0045190, -0.0022378, -0.0047444, -0.0021994, -0.0023196, 0.0025066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010586
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010407, upper bound: 0.0010850
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006425, 0.0013554, 0.0006247, 0.0014182, -0.0007667, 0.0007216
1: 0.9930440, 0.9948417, 0.9928857, 0.9948865, -0.0017322, 0.0018465
2: -0.0068178, -0.0046017, -0.0070934, -0.0045595, -0.0022583, 0.0024916
3: 0.0034834, 0.0042289, 0.0034647, 0.0042946, -0.0007370, 0.0006891
4: 0.0023740, 0.0038054, 0.0023533, 0.0040232, -0.0016492, 0.0014521
5: 0.0052657, 0.0070127, 0.0052220, 0.0071666, -0.0019009, 0.0017907
6: -0.0014876, -0.0006985, -0.0015833, -0.0006830, -0.0008046, 0.0008848
7: -0.0086616, -0.0074037, -0.0087725, -0.0073722, -0.0012894, 0.0013688
8: 0.0032402, 0.0072094, 0.0031428, 0.0075714, -0.0037518, 0.0034776
9: -0.0045190, -0.0022378, -0.0047201, -0.0021807, -0.0023383, 0.0024823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010591
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010407, upper bound: 0.0010916
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006333, 0.0013500, 0.0006349, 0.0014293, -0.0007872, 0.0007060
1: 0.9930574, 0.9948649, 0.9928576, 0.9948608, -0.0016914, 0.0018973
2: -0.0067945, -0.0045798, -0.0071421, -0.0045837, -0.0022107, 0.0025623
3: 0.0034737, 0.0042233, 0.0034754, 0.0043062, -0.0007573, 0.0006719
4: 0.0023632, 0.0037870, 0.0023652, 0.0040617, -0.0016985, 0.0014218
5: 0.0052430, 0.0069996, 0.0052471, 0.0071939, -0.0019509, 0.0017526
6: -0.0014795, -0.0006905, -0.0016002, -0.0006919, -0.0007876, 0.0009097
7: -0.0086522, -0.0073874, -0.0087921, -0.0073903, -0.0012619, 0.0014047
8: 0.0031896, 0.0071787, 0.0031986, 0.0076355, -0.0039041, 0.0033835
9: -0.0045020, -0.0022081, -0.0047557, -0.0022134, -0.0022886, 0.0025476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010121, upper bound: 0.0010845
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010330, upper bound: 0.0011006
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006333, 0.0013500, 0.0006177, 0.0014183, -0.0007759, 0.0007231
1: 0.9930574, 0.9948649, 0.9928854, 0.9949040, -0.0017327, 0.0018668
2: -0.0067945, -0.0045798, -0.0070937, -0.0045430, -0.0022515, 0.0025140
3: 0.0034737, 0.0042233, 0.0034574, 0.0042947, -0.0007440, 0.0006882
4: 0.0023632, 0.0037870, 0.0023452, 0.0040235, -0.0016603, 0.0014418
5: 0.0052430, 0.0069996, 0.0052049, 0.0071669, -0.0019239, 0.0017947
6: -0.0014795, -0.0006905, -0.0015834, -0.0006770, -0.0008025, 0.0008929
7: -0.0086522, -0.0073874, -0.0087726, -0.0073599, -0.0012923, 0.0013853
8: 0.0031896, 0.0071787, 0.0031048, 0.0075719, -0.0037698, 0.0034366
9: -0.0045020, -0.0022081, -0.0047204, -0.0021584, -0.0023436, 0.0025123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010121, upper bound: 0.0011040
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010330, upper bound: 0.0011154
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006260, 0.0014258, -0.0007904, 0.0007085
1: 0.9930742, 0.9948826, 0.9928665, 0.9948833, -0.0016994, 0.0019036
2: -0.0067652, -0.0045631, -0.0071269, -0.0045625, -0.0022027, 0.0025638
3: 0.0034663, 0.0042163, 0.0034660, 0.0043026, -0.0007599, 0.0006755
4: 0.0023551, 0.0037638, 0.0023548, 0.0040497, -0.0016947, 0.0014091
5: 0.0052257, 0.0069833, 0.0052251, 0.0071854, -0.0019596, 0.0017581
6: -0.0014693, -0.0006844, -0.0015949, -0.0006841, -0.0007852, 0.0009105
7: -0.0086404, -0.0073749, -0.0087860, -0.0073745, -0.0012659, 0.0014110
8: 0.0031512, 0.0071402, 0.0031498, 0.0076155, -0.0038484, 0.0034088
9: -0.0044806, -0.0021856, -0.0047446, -0.0021847, -0.0022959, 0.0025590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0011149
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0011149
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006202, 0.0014182, -0.0007830, 0.0007142
1: 0.9930742, 0.9948826, 0.9928854, 0.9948976, -0.0017154, 0.0018858
2: -0.0067652, -0.0045631, -0.0070936, -0.0045490, -0.0022162, 0.0025305
3: 0.0034663, 0.0042163, 0.0034601, 0.0042946, -0.0007527, 0.0006827
4: 0.0023551, 0.0037638, 0.0023481, 0.0040234, -0.0016683, 0.0014157
5: 0.0052257, 0.0069833, 0.0052111, 0.0071668, -0.0019410, 0.0017722
6: -0.0014693, -0.0006844, -0.0015833, -0.0006792, -0.0007901, 0.0008990
7: -0.0086404, -0.0073749, -0.0087726, -0.0073644, -0.0012761, 0.0013976
8: 0.0031512, 0.0071402, 0.0031186, 0.0075717, -0.0038113, 0.0034621
9: -0.0044806, -0.0021856, -0.0047203, -0.0021664, -0.0023142, 0.0025347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0011194
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0011304
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0006150, 0.0014183, 0.0006730, 0.0013796, -0.0007554, 0.0007362
1: 0.9928854, 0.9949108, 0.9929829, 0.9947646, -0.0017670, 0.0018145
2: -0.0070938, -0.0045365, -0.0069241, -0.0046738, -0.0024200, 0.0023876
3: 0.0034546, 0.0042947, 0.0035152, 0.0042542, -0.0007225, 0.0007029
4: 0.0023420, 0.0040236, 0.0024094, 0.0038894, -0.0015474, 0.0016142
5: 0.0051982, 0.0071669, 0.0053403, 0.0070721, -0.0018738, 0.0018265
6: -0.0015834, -0.0006746, -0.0015245, -0.0007249, -0.0008585, 0.0008499
7: -0.0087726, -0.0073551, -0.0087044, -0.0074575, -0.0013152, 0.0013493
8: 0.0030899, 0.0075720, 0.0034063, 0.0073490, -0.0036036, 0.0035431
9: -0.0047204, -0.0021497, -0.0045966, -0.0023352, -0.0023852, 0.0024469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010612, upper bound: 0.0010223
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010612, upper bound: 0.0010225
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0006108, 0.0014183, 0.0006615, 0.0013609, -0.0007409, 0.0007479
1: 0.9928852, 0.9949214, 0.9930301, 0.9947939, -0.0017981, 0.0017796
2: -0.0070940, -0.0045267, -0.0068421, -0.0046465, -0.0024474, 0.0023154
3: 0.0034502, 0.0042947, 0.0035032, 0.0042347, -0.0007084, 0.0007160
4: 0.0023372, 0.0040237, 0.0023960, 0.0038246, -0.0014874, 0.0016277
5: 0.0051880, 0.0071670, 0.0053121, 0.0070262, -0.0018382, 0.0018549
6: -0.0015835, -0.0006710, -0.0014960, -0.0007149, -0.0008686, 0.0008250
7: -0.0087727, -0.0073478, -0.0086714, -0.0074371, -0.0013356, 0.0013236
8: 0.0030673, 0.0075723, 0.0033434, 0.0072413, -0.0035678, 0.0035708
9: -0.0047205, -0.0021364, -0.0045368, -0.0022983, -0.0024222, 0.0024004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010851, upper bound: 0.0010405
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010851, upper bound: 0.0010414
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006305, 0.0014258, 0.0006425, 0.0013554, -0.0007159, 0.0007742
1: 0.9928665, 0.9948716, 0.9930440, 0.9948417, -0.0018650, 0.0017172
2: -0.0071267, -0.0045733, -0.0068178, -0.0046017, -0.0025249, 0.0022444
3: 0.0034708, 0.0043025, 0.0034834, 0.0042289, -0.0006827, 0.0007444
4: 0.0023601, 0.0040496, 0.0023740, 0.0038054, -0.0014453, 0.0016756
5: 0.0052363, 0.0071853, 0.0052657, 0.0070127, -0.0017763, 0.0019195
6: -0.0015948, -0.0006881, -0.0014876, -0.0006985, -0.0008963, 0.0007995
7: -0.0087859, -0.0073826, -0.0086616, -0.0074037, -0.0013822, 0.0012791
8: 0.0031747, 0.0076152, 0.0032402, 0.0072094, -0.0034244, 0.0037890
9: -0.0047444, -0.0021994, -0.0045190, -0.0022378, -0.0025066, 0.0023196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010586, upper bound: 0.0010223
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010846, upper bound: 0.0010407
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006247, 0.0014182, 0.0006425, 0.0013554, -0.0007216, 0.0007667
1: 0.9928857, 0.9948865, 0.9930440, 0.9948417, -0.0018465, 0.0017322
2: -0.0070934, -0.0045595, -0.0068178, -0.0046017, -0.0024916, 0.0022583
3: 0.0034647, 0.0042946, 0.0034834, 0.0042289, -0.0006891, 0.0007370
4: 0.0023533, 0.0040232, 0.0023740, 0.0038054, -0.0014521, 0.0016492
5: 0.0052220, 0.0071666, 0.0052657, 0.0070127, -0.0017907, 0.0019009
6: -0.0015833, -0.0006830, -0.0014876, -0.0006985, -0.0008848, 0.0008046
7: -0.0087725, -0.0073722, -0.0086616, -0.0074037, -0.0013688, 0.0012894
8: 0.0031428, 0.0075714, 0.0032402, 0.0072094, -0.0034776, 0.0037518
9: -0.0047201, -0.0021807, -0.0045190, -0.0022378, -0.0024823, 0.0023383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010586, upper bound: 0.0010225
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010846, upper bound: 0.0010432
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006349, 0.0014293, 0.0006333, 0.0013500, -0.0007060, 0.0007872
1: 0.9928576, 0.9948608, 0.9930574, 0.9948649, -0.0018973, 0.0016914
2: -0.0071421, -0.0045837, -0.0067945, -0.0045798, -0.0025623, 0.0022107
3: 0.0034754, 0.0043062, 0.0034737, 0.0042233, -0.0006719, 0.0007573
4: 0.0023652, 0.0040617, 0.0023632, 0.0037870, -0.0014218, 0.0016985
5: 0.0052471, 0.0071939, 0.0052430, 0.0069996, -0.0017526, 0.0019509
6: -0.0016002, -0.0006919, -0.0014795, -0.0006905, -0.0009097, 0.0007876
7: -0.0087921, -0.0073903, -0.0086522, -0.0073874, -0.0014047, 0.0012619
8: 0.0031986, 0.0076355, 0.0031896, 0.0071787, -0.0033835, 0.0039041
9: -0.0047557, -0.0022134, -0.0045020, -0.0022081, -0.0025476, 0.0022886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010605, upper bound: 0.0010121
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010869, upper bound: 0.0010330
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006177, 0.0014183, 0.0006333, 0.0013500, -0.0007231, 0.0007759
1: 0.9928854, 0.9949040, 0.9930574, 0.9948649, -0.0018668, 0.0017327
2: -0.0070937, -0.0045430, -0.0067945, -0.0045798, -0.0025140, 0.0022515
3: 0.0034574, 0.0042947, 0.0034737, 0.0042233, -0.0006882, 0.0007440
4: 0.0023452, 0.0040235, 0.0023632, 0.0037870, -0.0014418, 0.0016603
5: 0.0052049, 0.0071669, 0.0052430, 0.0069996, -0.0017947, 0.0019239
6: -0.0015834, -0.0006770, -0.0014795, -0.0006905, -0.0008929, 0.0008025
7: -0.0087726, -0.0073599, -0.0086522, -0.0073874, -0.0013853, 0.0012923
8: 0.0031048, 0.0075719, 0.0031896, 0.0071787, -0.0034366, 0.0037698
9: -0.0047204, -0.0021584, -0.0045020, -0.0022081, -0.0025123, 0.0023436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010605, upper bound: 0.0010370
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010869, upper bound: 0.0010553
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006260, 0.0014258, 0.0006262, 0.0013434, -0.0007085, 0.0007904
1: 0.9928665, 0.9948833, 0.9930742, 0.9948826, -0.0019036, 0.0016994
2: -0.0071269, -0.0045625, -0.0067652, -0.0045631, -0.0025638, 0.0022027
3: 0.0034660, 0.0043026, 0.0034663, 0.0042163, -0.0006755, 0.0007599
4: 0.0023548, 0.0040497, 0.0023551, 0.0037638, -0.0014091, 0.0016947
5: 0.0052251, 0.0071854, 0.0052257, 0.0069833, -0.0017581, 0.0019596
6: -0.0015949, -0.0006841, -0.0014693, -0.0006844, -0.0009105, 0.0007852
7: -0.0087860, -0.0073745, -0.0086404, -0.0073749, -0.0014110, 0.0012659
8: 0.0031498, 0.0076155, 0.0031512, 0.0071402, -0.0034088, 0.0038484
9: -0.0047446, -0.0021847, -0.0044806, -0.0021856, -0.0025590, 0.0022959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011012, upper bound: 0.0010477
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011012, upper bound: 0.0010711
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006202, 0.0014182, 0.0006262, 0.0013434, -0.0007142, 0.0007830
1: 0.9928854, 0.9948976, 0.9930742, 0.9948826, -0.0018858, 0.0017154
2: -0.0070936, -0.0045490, -0.0067652, -0.0045631, -0.0025305, 0.0022162
3: 0.0034601, 0.0042946, 0.0034663, 0.0042163, -0.0006827, 0.0007527
4: 0.0023481, 0.0040234, 0.0023551, 0.0037638, -0.0014157, 0.0016683
5: 0.0052111, 0.0071668, 0.0052257, 0.0069833, -0.0017722, 0.0019410
6: -0.0015833, -0.0006792, -0.0014693, -0.0006844, -0.0008990, 0.0007901
7: -0.0087726, -0.0073644, -0.0086404, -0.0073749, -0.0013976, 0.0012761
8: 0.0031186, 0.0075717, 0.0031512, 0.0071402, -0.0034621, 0.0038113
9: -0.0047203, -0.0021664, -0.0044806, -0.0021856, -0.0025347, 0.0023142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011012, upper bound: 0.0010515
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011012, upper bound: 0.0010725
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006391, 0.0014293, 0.0006531, 0.0014447, -0.0007960, 0.0007666
1: 0.9928576, 0.9948503, 0.9928188, 0.9948150, -0.0018385, 0.0019124
2: -0.0071419, -0.0045935, -0.0072096, -0.0046267, -0.0025152, 0.0026161
3: 0.0034797, 0.0043062, 0.0034944, 0.0043223, -0.0007612, 0.0007304
4: 0.0023700, 0.0040616, 0.0023863, 0.0041151, -0.0017451, 0.0016753
5: 0.0052572, 0.0071938, 0.0052916, 0.0072316, -0.0019744, 0.0019022
6: -0.0016001, -0.0006955, -0.0016236, -0.0007076, -0.0008925, 0.0009281
7: -0.0087920, -0.0073976, -0.0088192, -0.0074223, -0.0013697, 0.0014216
8: 0.0032212, 0.0076352, 0.0032977, 0.0077242, -0.0037103, 0.0035832
9: -0.0047555, -0.0022267, -0.0048049, -0.0022716, -0.0024840, 0.0025782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010883, upper bound: 0.0010430
time: 1.01 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010883, upper bound: 0.0010477
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006349, 0.0014293, 0.0006414, 0.0014257, -0.0007811, 0.0007783
1: 0.9928576, 0.9948608, 0.9928667, 0.9948444, -0.0018684, 0.0018749
2: -0.0071421, -0.0045837, -0.0071262, -0.0045991, -0.0025430, 0.0025425
3: 0.0034754, 0.0043062, 0.0034822, 0.0043024, -0.0007461, 0.0007427
4: 0.0023652, 0.0040617, 0.0023727, 0.0040492, -0.0016840, 0.0016890
5: 0.0052471, 0.0071939, 0.0052630, 0.0071850, -0.0019379, 0.0019309
6: -0.0016002, -0.0006919, -0.0015947, -0.0006975, -0.0009027, 0.0009028
7: -0.0087921, -0.0073903, -0.0087857, -0.0074017, -0.0013904, 0.0013954
8: 0.0031986, 0.0076355, 0.0032340, 0.0076146, -0.0036781, 0.0036152
9: -0.0047557, -0.0022134, -0.0047441, -0.0022342, -0.0025215, 0.0025306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011048, upper bound: 0.0010570
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011048, upper bound: 0.0010620
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006177, 0.0014183, 0.0006551, 0.0014364, -0.0008093, 0.0007536
1: 0.9928854, 0.9949040, 0.9928396, 0.9948098, -0.0018068, 0.0019469
2: -0.0070937, -0.0045430, -0.0071734, -0.0046315, -0.0024623, 0.0026304
3: 0.0034574, 0.0042947, 0.0034965, 0.0043137, -0.0007757, 0.0007177
4: 0.0023452, 0.0040235, 0.0023886, 0.0040865, -0.0017413, 0.0016349
5: 0.0052049, 0.0071669, 0.0052965, 0.0072114, -0.0020064, 0.0018703
6: -0.0015834, -0.0006770, -0.0016110, -0.0007094, -0.0008740, 0.0009340
7: -0.0087726, -0.0073599, -0.0088047, -0.0074259, -0.0013467, 0.0014447
8: 0.0031048, 0.0075719, 0.0033087, 0.0076766, -0.0038641, 0.0035532
9: -0.0047204, -0.0021584, -0.0047785, -0.0022780, -0.0024424, 0.0026201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010665, upper bound: 0.0010559
time: 1.03 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010925, upper bound: 0.0010684
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006177, 0.0014183, 0.0006375, 0.0014257, -0.0007983, 0.0007712
1: 0.9928854, 0.9949040, 0.9928666, 0.9948542, -0.0018493, 0.0019165
2: -0.0070937, -0.0045430, -0.0071265, -0.0045897, -0.0025040, 0.0025835
3: 0.0034574, 0.0042947, 0.0034781, 0.0043025, -0.0007624, 0.0007344
4: 0.0023452, 0.0040235, 0.0023681, 0.0040494, -0.0017042, 0.0016554
5: 0.0052049, 0.0071669, 0.0052533, 0.0071851, -0.0019802, 0.0019136
6: -0.0015834, -0.0006770, -0.0015947, -0.0006941, -0.0008893, 0.0009177
7: -0.0087726, -0.0073599, -0.0087858, -0.0073948, -0.0013779, 0.0014259
8: 0.0031048, 0.0075719, 0.0032124, 0.0076149, -0.0037375, 0.0036158
9: -0.0047204, -0.0021584, -0.0047442, -0.0022215, -0.0024989, 0.0025859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011073, upper bound: 0.0010953
time: 0.95 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011073, upper bound: 0.0010966
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0006551, 0.0014364, 0.0006247, 0.0014182, -0.0007535, 0.0008021
1: 0.9928396, 0.9948098, 0.9928857, 0.9948865, -0.0019290, 0.0018066
2: -0.0071734, -0.0046315, -0.0070934, -0.0045595, -0.0026139, 0.0024619
3: 0.0034965, 0.0043137, 0.0034647, 0.0042946, -0.0007176, 0.0007685
4: 0.0023886, 0.0040865, 0.0023533, 0.0040232, -0.0016346, 0.0017332
5: 0.0052965, 0.0072114, 0.0052220, 0.0071666, -0.0018701, 0.0019894
6: -0.0016110, -0.0007094, -0.0015833, -0.0006830, -0.0009280, 0.0008739
7: -0.0088047, -0.0074259, -0.0087725, -0.0073722, -0.0014324, 0.0013466
8: 0.0033087, 0.0076766, 0.0031428, 0.0075714, -0.0035048, 0.0038046
9: -0.0047785, -0.0022780, -0.0047201, -0.0021807, -0.0025978, 0.0024421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010860, upper bound: 0.0010430
time: 1.09 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011029, upper bound: 0.0010571
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0006489, 0.0014292, 0.0006247, 0.0014182, -0.0007598, 0.0007950
1: 0.9928579, 0.9948256, 0.9928857, 0.9948865, -0.0019128, 0.0018238
2: -0.0071416, -0.0046168, -0.0070934, -0.0045595, -0.0025821, 0.0024766
3: 0.0034900, 0.0043061, 0.0034647, 0.0042946, -0.0007256, 0.0007622
4: 0.0023814, 0.0040613, 0.0023533, 0.0040232, -0.0016418, 0.0017080
5: 0.0052813, 0.0071936, 0.0052220, 0.0071666, -0.0018853, 0.0019716
6: -0.0016000, -0.0007040, -0.0015833, -0.0006830, -0.0009169, 0.0008793
7: -0.0087919, -0.0074150, -0.0087725, -0.0073722, -0.0014196, 0.0013575
8: 0.0032749, 0.0076348, 0.0031428, 0.0075714, -0.0035829, 0.0037759
9: -0.0047553, -0.0022582, -0.0047201, -0.0021807, -0.0025746, 0.0024619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010860, upper bound: 0.0010508
time: 1.00 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011029, upper bound: 0.0010570
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006202, 0.0014182, -0.0007711, 0.0007959
1: 0.9928666, 0.9948542, 0.9928854, 0.9948976, -0.0019136, 0.0018492
2: -0.0071265, -0.0045897, -0.0070936, -0.0045490, -0.0025775, 0.0025039
3: 0.0034781, 0.0043025, 0.0034601, 0.0042946, -0.0007344, 0.0007626
4: 0.0023681, 0.0040494, 0.0023481, 0.0040234, -0.0016553, 0.0017013
5: 0.0052533, 0.0071851, 0.0052111, 0.0071668, -0.0019135, 0.0019740
6: -0.0015947, -0.0006941, -0.0015833, -0.0006792, -0.0009156, 0.0008892
7: -0.0087858, -0.0073948, -0.0087726, -0.0073644, -0.0014214, 0.0013778
8: 0.0032124, 0.0076149, 0.0031186, 0.0075717, -0.0035672, 0.0037838
9: -0.0047442, -0.0022215, -0.0047203, -0.0021664, -0.0025778, 0.0024987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011072, upper bound: 0.0010841
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011072, upper bound: 0.0010953
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006202, 0.0014182, -0.0007769, 0.0007884
1: 0.9928859, 0.9948689, 0.9928854, 0.9948976, -0.0018971, 0.0018653
2: -0.0070932, -0.0045761, -0.0070936, -0.0045490, -0.0025442, 0.0025174
3: 0.0034720, 0.0042945, 0.0034601, 0.0042946, -0.0007419, 0.0007560
4: 0.0023615, 0.0040230, 0.0023481, 0.0040234, -0.0016619, 0.0016749
5: 0.0052392, 0.0071665, 0.0052111, 0.0071668, -0.0019275, 0.0019554
6: -0.0015832, -0.0006891, -0.0015833, -0.0006792, -0.0009040, 0.0008942
7: -0.0087724, -0.0073846, -0.0087726, -0.0073644, -0.0014080, 0.0013879
8: 0.0031812, 0.0075711, 0.0031186, 0.0075717, -0.0036369, 0.0037600
9: -0.0047199, -0.0022032, -0.0047203, -0.0021664, -0.0025535, 0.0025171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011072, upper bound: 0.0010910
time: 1.04 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011072, upper bound: 0.0010953
time: 1.04 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.39 seconds
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010234, upper bound: 0.0010100
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010406, upper bound: 0.0010299
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010234, upper bound: 0.0010121
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010406, upper bound: 0.0010346
IS_A1_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010234
IS_A1_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010414
IS_A1_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010377
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010561
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010585
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010603
IS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010725
IS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0010734
IS_A1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010612
IS_A1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010619
IS_A1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010405, upper bound: 0.0010851
IS_A1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010405, upper bound: 0.0010874
IS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010586
IS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010407, upper bound: 0.0010850
IS_A1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010223, upper bound: 0.0010591
IS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010407, upper bound: 0.0010916
IS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010121, upper bound: 0.0010845
IS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010330, upper bound: 0.0011006
IS_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010121, upper bound: 0.0011040
IS_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010330, upper bound: 0.0011154
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0011149
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0011149
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0011194
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010477, upper bound: 0.0011304
IS_A2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010612, upper bound: 0.0010223
IS_A2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010612, upper bound: 0.0010225
IS_A2_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010851, upper bound: 0.0010405
IS_A2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010851, upper bound: 0.0010414
IS_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010586, upper bound: 0.0010223
IS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010846, upper bound: 0.0010407
IS_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010586, upper bound: 0.0010225
IS_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010846, upper bound: 0.0010432
IS_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010605, upper bound: 0.0010121
IS_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010869, upper bound: 0.0010330
IS_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010605, upper bound: 0.0010370
IS_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010869, upper bound: 0.0010553
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011012, upper bound: 0.0010477
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011012, upper bound: 0.0010711
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011012, upper bound: 0.0010515
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011012, upper bound: 0.0010725
IS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010883, upper bound: 0.0010430
IS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010883, upper bound: 0.0010477
IS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011048, upper bound: 0.0010570
IS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011048, upper bound: 0.0010620
IS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010665, upper bound: 0.0010559
IS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010925, upper bound: 0.0010684
IS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011073, upper bound: 0.0010953
IS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011073, upper bound: 0.0010966
IS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010860, upper bound: 0.0010430
IS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011029, upper bound: 0.0010571
IS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0010860, upper bound: 0.0010508
IS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011029, upper bound: 0.0010570
IS_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011072, upper bound: 0.0010841
IS_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011072, upper bound: 0.0010953
IS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011072, upper bound: 0.0010910
IS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 1, lower bound: -0.0011072, upper bound: 0.0010953

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006237, 0.0013434, 0.0006333, 0.0013500, -0.0007169, 0.0007009
1: 0.9930744, 0.9948889, 0.9930574, 0.9948649, -0.0016760, 0.0017143
2: -0.0067651, -0.0045572, -0.0067945, -0.0045798, -0.0021853, 0.0022372
3: 0.0034637, 0.0042163, 0.0034737, 0.0042233, -0.0006794, 0.0006642
4: 0.0023522, 0.0037638, 0.0023632, 0.0037870, -0.0014348, 0.0014005
5: 0.0052196, 0.0069832, 0.0052430, 0.0069996, -0.0017800, 0.0017402
6: -0.0014693, -0.0006822, -0.0014795, -0.0006905, -0.0007789, 0.0007973
7: -0.0086404, -0.0073705, -0.0086522, -0.0073874, -0.0012531, 0.0012817
8: 0.0031376, 0.0071401, 0.0031896, 0.0071787, -0.0033193, 0.0033458
9: -0.0044806, -0.0021776, -0.0045020, -0.0022081, -0.0022725, 0.0023244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010556, upper bound: 0.0010556
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010556, upper bound: 0.0010561
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006333, 0.0013500, 0.0006425, 0.0013554, -0.0007131, 0.0006983
1: 0.9930574, 0.9948649, 0.9930440, 0.9948417, -0.0016716, 0.0017085
2: -0.0067945, -0.0045798, -0.0068178, -0.0046017, -0.0021927, 0.0022380
3: 0.0034737, 0.0042233, 0.0034834, 0.0042289, -0.0006786, 0.0006634
4: 0.0023632, 0.0037870, 0.0023740, 0.0038054, -0.0014422, 0.0014130
5: 0.0052430, 0.0069996, 0.0052657, 0.0070127, -0.0017697, 0.0017339
6: -0.0014795, -0.0006905, -0.0014876, -0.0006985, -0.0007810, 0.0007972
7: -0.0086522, -0.0073874, -0.0086616, -0.0074037, -0.0012485, 0.0012743
8: 0.0031896, 0.0071787, 0.0032402, 0.0072094, -0.0034182, 0.0033320
9: -0.0045020, -0.0022081, -0.0045190, -0.0022378, -0.0022642, 0.0023109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010234
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010408
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006425, 0.0013554, -0.0007200, 0.0006917
1: 0.9930742, 0.9948826, 0.9930440, 0.9948417, -0.0016548, 0.0017260
2: -0.0067652, -0.0045631, -0.0068178, -0.0046017, -0.0021634, 0.0022547
3: 0.0034663, 0.0042163, 0.0034834, 0.0042289, -0.0006858, 0.0006565
4: 0.0023551, 0.0037638, 0.0023740, 0.0038054, -0.0014504, 0.0013898
5: 0.0052257, 0.0069833, 0.0052657, 0.0070127, -0.0017869, 0.0017175
6: -0.0014693, -0.0006844, -0.0014876, -0.0006985, -0.0007708, 0.0008033
7: -0.0086404, -0.0073749, -0.0086616, -0.0074037, -0.0012367, 0.0012867
8: 0.0031512, 0.0071402, 0.0032402, 0.0072094, -0.0034611, 0.0032761
9: -0.0044806, -0.0021856, -0.0045190, -0.0022378, -0.0022428, 0.0023335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010234
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010432
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006333, 0.0013500, 0.0006262, 0.0013434, -0.0007009, 0.0007145
1: 0.9930574, 0.9948649, 0.9930742, 0.9948826, -0.0017101, 0.0016761
2: -0.0067945, -0.0045798, -0.0067652, -0.0045631, -0.0022313, 0.0021854
3: 0.0034737, 0.0042233, 0.0034663, 0.0042163, -0.0006642, 0.0006789
4: 0.0023632, 0.0037870, 0.0023551, 0.0037638, -0.0014006, 0.0014319
5: 0.0052430, 0.0069996, 0.0052257, 0.0069833, -0.0017403, 0.0017739
6: -0.0014795, -0.0006905, -0.0014693, -0.0006844, -0.0007951, 0.0007789
7: -0.0086522, -0.0073874, -0.0086404, -0.0073749, -0.0012773, 0.0012531
8: 0.0031896, 0.0071787, 0.0031512, 0.0071402, -0.0032858, 0.0033828
9: -0.0045020, -0.0022081, -0.0044806, -0.0021856, -0.0023164, 0.0022725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0010368
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010548, upper bound: 0.0010556
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006262, 0.0013434, -0.0007079, 0.0007079
1: 0.9930742, 0.9948826, 0.9930742, 0.9948826, -0.0016942, 0.0016942
2: -0.0067652, -0.0045631, -0.0067652, -0.0045631, -0.0022020, 0.0022020
3: 0.0034663, 0.0042163, 0.0034663, 0.0042163, -0.0006722, 0.0006722
4: 0.0023551, 0.0037638, 0.0023551, 0.0037638, -0.0014088, 0.0014088
5: 0.0052257, 0.0069833, 0.0052257, 0.0069833, -0.0017575, 0.0017575
6: -0.0014693, -0.0006844, -0.0014693, -0.0006844, -0.0007850, 0.0007850
7: -0.0086404, -0.0073749, -0.0086404, -0.0073749, -0.0012655, 0.0012655
8: 0.0031512, 0.0071402, 0.0031512, 0.0071402, -0.0033278, 0.0033278
9: -0.0044806, -0.0021856, -0.0044806, -0.0021856, -0.0022951, 0.0022951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010350, upper bound: 0.0010377
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010548, upper bound: 0.0010570
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006730, 0.0013796, 0.0006347, 0.0014257, -0.0007436, 0.0007357
1: 0.9929829, 0.9947646, 0.9928665, 0.9948614, -0.0017649, 0.0017848
2: -0.0069241, -0.0046738, -0.0071265, -0.0045831, -0.0023410, 0.0024527
3: 0.0035152, 0.0042542, 0.0034751, 0.0043025, -0.0007101, 0.0007019
4: 0.0024094, 0.0038894, 0.0023649, 0.0040494, -0.0016400, 0.0015246
5: 0.0053403, 0.0070721, 0.0052464, 0.0071851, -0.0018448, 0.0018257
6: -0.0015245, -0.0007249, -0.0015948, -0.0006917, -0.0008328, 0.0008699
7: -0.0087044, -0.0074575, -0.0087858, -0.0073898, -0.0013146, 0.0013284
8: 0.0034063, 0.0073490, 0.0031972, 0.0076150, -0.0035363, 0.0034675
9: -0.0045966, -0.0023352, -0.0047443, -0.0022125, -0.0023840, 0.0024090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010597
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010612
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006730, 0.0013796, 0.0006289, 0.0014182, -0.0007361, 0.0007414
1: 0.9929829, 0.9947646, 0.9928858, 0.9948760, -0.0017797, 0.0017666
2: -0.0069241, -0.0046738, -0.0070932, -0.0045694, -0.0023547, 0.0024194
3: 0.0035152, 0.0042542, 0.0034691, 0.0042945, -0.0007028, 0.0007085
4: 0.0024094, 0.0038894, 0.0023581, 0.0040231, -0.0016137, 0.0015313
5: 0.0053403, 0.0070721, 0.0052322, 0.0071665, -0.0018262, 0.0018399
6: -0.0015245, -0.0007249, -0.0015832, -0.0006866, -0.0008379, 0.0008583
7: -0.0087044, -0.0074575, -0.0087724, -0.0073796, -0.0013248, 0.0013150
8: 0.0034063, 0.0073490, 0.0031656, 0.0075712, -0.0034981, 0.0035143
9: -0.0045966, -0.0023352, -0.0047200, -0.0021940, -0.0024026, 0.0023847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010605
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010619
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006615, 0.0013609, 0.0006305, 0.0014258, -0.0007553, 0.0007213
1: 0.9930301, 0.9947939, 0.9928665, 0.9948716, -0.0017298, 0.0018161
2: -0.0068421, -0.0046465, -0.0071267, -0.0045733, -0.0022687, 0.0024802
3: 0.0035032, 0.0042347, 0.0034708, 0.0043025, -0.0007234, 0.0006876
4: 0.0023960, 0.0038246, 0.0023601, 0.0040496, -0.0016536, 0.0014645
5: 0.0053121, 0.0070262, 0.0052363, 0.0071853, -0.0018732, 0.0017899
6: -0.0014960, -0.0007149, -0.0015948, -0.0006881, -0.0008079, 0.0008800
7: -0.0086714, -0.0074371, -0.0087859, -0.0073826, -0.0012888, 0.0013488
8: 0.0033434, 0.0072413, 0.0031747, 0.0076152, -0.0035748, 0.0034315
9: -0.0045368, -0.0022983, -0.0047444, -0.0021994, -0.0023374, 0.0024461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010845
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010851
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006615, 0.0013609, 0.0006247, 0.0014182, -0.0007478, 0.0007269
1: 0.9930301, 0.9947939, 0.9928857, 0.9948865, -0.0017448, 0.0017978
2: -0.0068421, -0.0046465, -0.0070934, -0.0045595, -0.0022826, 0.0024469
3: 0.0035032, 0.0042347, 0.0034647, 0.0042946, -0.0007159, 0.0006943
4: 0.0023960, 0.0038246, 0.0023533, 0.0040232, -0.0016273, 0.0014713
5: 0.0053121, 0.0070262, 0.0052220, 0.0071666, -0.0018545, 0.0018042
6: -0.0014960, -0.0007149, -0.0015833, -0.0006830, -0.0008130, 0.0008684
7: -0.0086714, -0.0074371, -0.0087725, -0.0073722, -0.0012991, 0.0013354
8: 0.0033434, 0.0072413, 0.0031428, 0.0075714, -0.0035294, 0.0034794
9: -0.0045368, -0.0022983, -0.0047201, -0.0021807, -0.0023561, 0.0024218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010869
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010874
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006660, 0.0013728, 0.0006347, 0.0014257, -0.0007506, 0.0007292
1: 0.9930000, 0.9947824, 0.9928665, 0.9948614, -0.0017506, 0.0018038
2: -0.0068944, -0.0046572, -0.0071265, -0.0045831, -0.0023113, 0.0024693
3: 0.0035079, 0.0042471, 0.0034751, 0.0043025, -0.0007190, 0.0006961
4: 0.0024012, 0.0038660, 0.0023649, 0.0040494, -0.0016482, 0.0015011
5: 0.0053231, 0.0070555, 0.0052464, 0.0071851, -0.0018620, 0.0018091
6: -0.0015142, -0.0007188, -0.0015948, -0.0006917, -0.0008225, 0.0008760
7: -0.0086924, -0.0074451, -0.0087858, -0.0073898, -0.0013026, 0.0013407
8: 0.0033680, 0.0073100, 0.0031972, 0.0076150, -0.0036149, 0.0034543
9: -0.0045749, -0.0023128, -0.0047443, -0.0022125, -0.0023624, 0.0024315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010571
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010586
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006532, 0.0013553, 0.0006305, 0.0014258, -0.0007636, 0.0007158
1: 0.9930442, 0.9948147, 0.9928665, 0.9948716, -0.0017171, 0.0018380
2: -0.0068175, -0.0046269, -0.0071267, -0.0045733, -0.0022442, 0.0024998
3: 0.0034945, 0.0042288, 0.0034708, 0.0043025, -0.0007330, 0.0006826
4: 0.0023863, 0.0038052, 0.0023601, 0.0040496, -0.0016632, 0.0014451
5: 0.0052917, 0.0070125, 0.0052363, 0.0071853, -0.0018935, 0.0017762
6: -0.0014875, -0.0007077, -0.0015948, -0.0006881, -0.0007994, 0.0008872
7: -0.0086615, -0.0074224, -0.0087859, -0.0073826, -0.0012790, 0.0013634
8: 0.0032981, 0.0072090, 0.0031747, 0.0076152, -0.0036562, 0.0034232
9: -0.0045188, -0.0022718, -0.0047444, -0.0021994, -0.0023194, 0.0024727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010848
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010848
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006660, 0.0013728, 0.0006289, 0.0014182, -0.0007432, 0.0007349
1: 0.9930000, 0.9947824, 0.9928858, 0.9948760, -0.0017660, 0.0017859
2: -0.0068944, -0.0046572, -0.0070932, -0.0045694, -0.0023250, 0.0024360
3: 0.0035079, 0.0042471, 0.0034691, 0.0042945, -0.0007115, 0.0007028
4: 0.0024012, 0.0038660, 0.0023581, 0.0040231, -0.0016219, 0.0015078
5: 0.0053231, 0.0070555, 0.0052322, 0.0071665, -0.0018434, 0.0018233
6: -0.0015142, -0.0007188, -0.0015832, -0.0006866, -0.0008276, 0.0008644
7: -0.0086924, -0.0074451, -0.0087724, -0.0073796, -0.0013129, 0.0013273
8: 0.0033680, 0.0073100, 0.0031656, 0.0075712, -0.0035886, 0.0035155
9: -0.0045749, -0.0023128, -0.0047200, -0.0021940, -0.0023809, 0.0024072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010106, upper bound: 0.0010574
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010106, upper bound: 0.0010591
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006532, 0.0013553, 0.0006247, 0.0014182, -0.0007561, 0.0007215
1: 0.9930442, 0.9948147, 0.9928857, 0.9948865, -0.0017321, 0.0018195
2: -0.0068175, -0.0046269, -0.0070934, -0.0045595, -0.0022580, 0.0024665
3: 0.0034945, 0.0042288, 0.0034647, 0.0042946, -0.0007256, 0.0006890
4: 0.0023863, 0.0038052, 0.0023533, 0.0040232, -0.0016369, 0.0014519
5: 0.0052917, 0.0070125, 0.0052220, 0.0071666, -0.0018749, 0.0017905
6: -0.0014875, -0.0007077, -0.0015833, -0.0006830, -0.0008045, 0.0008756
7: -0.0086615, -0.0074224, -0.0087725, -0.0073722, -0.0012893, 0.0013500
8: 0.0032981, 0.0072090, 0.0031428, 0.0075714, -0.0036351, 0.0034764
9: -0.0045188, -0.0022718, -0.0047201, -0.0021807, -0.0023381, 0.0024483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010341, upper bound: 0.0010913
time: 1.47 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010341, upper bound: 0.0010916
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006561, 0.0013691, 0.0006391, 0.0014293, -0.0007641, 0.0007207
1: 0.9930096, 0.9948073, 0.9928576, 0.9948503, -0.0017270, 0.0018381
2: -0.0068779, -0.0046340, -0.0071419, -0.0045935, -0.0022843, 0.0025080
3: 0.0034976, 0.0042432, 0.0034797, 0.0043062, -0.0007326, 0.0006864
4: 0.0023898, 0.0038529, 0.0023700, 0.0040616, -0.0016718, 0.0014829
5: 0.0052991, 0.0070462, 0.0052572, 0.0071938, -0.0018947, 0.0017890
6: -0.0015085, -0.0007103, -0.0016001, -0.0006955, -0.0008130, 0.0008898
7: -0.0086858, -0.0074277, -0.0087920, -0.0073976, -0.0012882, 0.0013643
8: 0.0033144, 0.0072883, 0.0032212, 0.0076352, -0.0037382, 0.0034166
9: -0.0045629, -0.0022813, -0.0047555, -0.0022267, -0.0023362, 0.0024742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010834
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010845
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006440, 0.0013500, 0.0006349, 0.0014293, -0.0007764, 0.0007059
1: 0.9930576, 0.9948378, 0.9928576, 0.9948608, -0.0016912, 0.0018697
2: -0.0067942, -0.0046051, -0.0071421, -0.0045837, -0.0022105, 0.0025370
3: 0.0034849, 0.0042232, 0.0034754, 0.0043062, -0.0007456, 0.0006719
4: 0.0023757, 0.0037868, 0.0023652, 0.0040617, -0.0016861, 0.0014216
5: 0.0052692, 0.0069995, 0.0052471, 0.0071939, -0.0019246, 0.0017524
6: -0.0014794, -0.0006997, -0.0016002, -0.0006919, -0.0007875, 0.0009005
7: -0.0086521, -0.0074062, -0.0087921, -0.0073903, -0.0012618, 0.0013858
8: 0.0032480, 0.0071784, 0.0031986, 0.0076355, -0.0037700, 0.0033823
9: -0.0045018, -0.0022424, -0.0047557, -0.0022134, -0.0022884, 0.0025133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010996
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0011006
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006561, 0.0013691, 0.0006219, 0.0014182, -0.0007529, 0.0007378
1: 0.9930096, 0.9948073, 0.9928856, 0.9948934, -0.0017681, 0.0018076
2: -0.0068779, -0.0046340, -0.0070936, -0.0045529, -0.0023250, 0.0024596
3: 0.0034976, 0.0042432, 0.0034618, 0.0042946, -0.0007191, 0.0007026
4: 0.0023898, 0.0038529, 0.0023501, 0.0040234, -0.0016336, 0.0015028
5: 0.0052991, 0.0070462, 0.0052152, 0.0071667, -0.0018677, 0.0018311
6: -0.0015085, -0.0007103, -0.0015833, -0.0006806, -0.0008278, 0.0008731
7: -0.0086858, -0.0074277, -0.0087726, -0.0073673, -0.0013185, 0.0013448
8: 0.0033144, 0.0072883, 0.0031276, 0.0075717, -0.0036078, 0.0034728
9: -0.0045629, -0.0022813, -0.0047202, -0.0021718, -0.0023911, 0.0024389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010350, upper bound: 0.0011040
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010350, upper bound: 0.0011040
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006440, 0.0013500, 0.0006177, 0.0014183, -0.0007652, 0.0007231
1: 0.9930576, 0.9948378, 0.9928854, 0.9949040, -0.0017325, 0.0018394
2: -0.0067942, -0.0046051, -0.0070937, -0.0045430, -0.0022512, 0.0024886
3: 0.0034849, 0.0042232, 0.0034574, 0.0042947, -0.0007324, 0.0006881
4: 0.0023757, 0.0037868, 0.0023452, 0.0040235, -0.0016478, 0.0014416
5: 0.0052692, 0.0069995, 0.0052049, 0.0071669, -0.0018976, 0.0017946
6: -0.0014794, -0.0006997, -0.0015834, -0.0006770, -0.0008024, 0.0008837
7: -0.0086521, -0.0074062, -0.0087726, -0.0073599, -0.0012922, 0.0013664
8: 0.0032480, 0.0071784, 0.0031048, 0.0075719, -0.0036383, 0.0034353
9: -0.0045018, -0.0022424, -0.0047204, -0.0021584, -0.0023435, 0.0024780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010535, upper bound: 0.0011154
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010535, upper bound: 0.0011154
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006551, 0.0014364, -0.0008013, 0.0006793
1: 0.9930742, 0.9948826, 0.9928396, 0.9948098, -0.0016251, 0.0019336
2: -0.0067652, -0.0045631, -0.0071734, -0.0046315, -0.0021337, 0.0026103
3: 0.0034663, 0.0042163, 0.0034965, 0.0043137, -0.0007732, 0.0006448
4: 0.0023551, 0.0037638, 0.0023886, 0.0040865, -0.0017314, 0.0013752
5: 0.0052257, 0.0069833, 0.0052965, 0.0072114, -0.0019856, 0.0016868
6: -0.0014693, -0.0006844, -0.0016110, -0.0007094, -0.0007600, 0.0009267
7: -0.0086404, -0.0073749, -0.0088047, -0.0074259, -0.0012146, 0.0014298
8: 0.0031512, 0.0071402, 0.0033087, 0.0076766, -0.0039745, 0.0032394
9: -0.0044806, -0.0021856, -0.0047785, -0.0022780, -0.0022026, 0.0025929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010787
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010975
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006375, 0.0014257, -0.0007903, 0.0006968
1: 0.9930742, 0.9948826, 0.9928666, 0.9948542, -0.0016678, 0.0019033
2: -0.0067652, -0.0045631, -0.0071265, -0.0045897, -0.0021755, 0.0025634
3: 0.0034663, 0.0042163, 0.0034781, 0.0043025, -0.0007599, 0.0006614
4: 0.0023551, 0.0037638, 0.0023681, 0.0040494, -0.0016943, 0.0013957
5: 0.0052257, 0.0069833, 0.0052533, 0.0071851, -0.0019594, 0.0017300
6: -0.0014693, -0.0006844, -0.0015947, -0.0006941, -0.0007753, 0.0009104
7: -0.0086404, -0.0073749, -0.0087858, -0.0073948, -0.0012457, 0.0014109
8: 0.0031512, 0.0071402, 0.0032124, 0.0076149, -0.0038469, 0.0032915
9: -0.0044806, -0.0021856, -0.0047442, -0.0022215, -0.0022591, 0.0025587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010977
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0011105
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006489, 0.0014292, -0.0007942, 0.0006855
1: 0.9930742, 0.9948826, 0.9928579, 0.9948256, -0.0016420, 0.0019166
2: -0.0067652, -0.0045631, -0.0071416, -0.0046168, -0.0021483, 0.0025785
3: 0.0034663, 0.0042163, 0.0034900, 0.0043061, -0.0007660, 0.0006523
4: 0.0023551, 0.0037638, 0.0023814, 0.0040613, -0.0017063, 0.0013824
5: 0.0052257, 0.0069833, 0.0052813, 0.0071936, -0.0019678, 0.0017019
6: -0.0014693, -0.0006844, -0.0016000, -0.0007040, -0.0007653, 0.0009156
7: -0.0086404, -0.0073749, -0.0087919, -0.0074150, -0.0012255, 0.0014169
8: 0.0031512, 0.0071402, 0.0032749, 0.0076348, -0.0039440, 0.0032907
9: -0.0044806, -0.0021856, -0.0047553, -0.0022582, -0.0022225, 0.0025697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010107, upper bound: 0.0010792
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010343, upper bound: 0.0011021
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0006262, 0.0013434, 0.0006317, 0.0014181, -0.0007829, 0.0007025
1: 0.9930742, 0.9948826, 0.9928859, 0.9948689, -0.0016836, 0.0018855
2: -0.0067652, -0.0045631, -0.0070932, -0.0045761, -0.0021890, 0.0025300
3: 0.0034663, 0.0042163, 0.0034720, 0.0042945, -0.0007526, 0.0006686
4: 0.0023551, 0.0037638, 0.0023615, 0.0040230, -0.0016680, 0.0014024
5: 0.0052257, 0.0069833, 0.0052392, 0.0071665, -0.0019408, 0.0017440
6: -0.0014693, -0.0006844, -0.0015832, -0.0006891, -0.0007802, 0.0008988
7: -0.0086404, -0.0073749, -0.0087724, -0.0073846, -0.0012558, 0.0013975
8: 0.0031512, 0.0071402, 0.0031812, 0.0075711, -0.0038098, 0.0033426
9: -0.0044806, -0.0021856, -0.0047199, -0.0022032, -0.0022774, 0.0025344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010107, upper bound: 0.0010978
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010343, upper bound: 0.0011021
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006347, 0.0014257, 0.0006730, 0.0013796, -0.0007357, 0.0007436
1: 0.9928665, 0.9948614, 0.9929829, 0.9947646, -0.0017848, 0.0017649
2: -0.0071265, -0.0045831, -0.0069241, -0.0046738, -0.0024527, 0.0023410
3: 0.0034751, 0.0043025, 0.0035152, 0.0042542, -0.0007019, 0.0007101
4: 0.0023649, 0.0040494, 0.0024094, 0.0038894, -0.0015246, 0.0016400
5: 0.0052464, 0.0071851, 0.0053403, 0.0070721, -0.0018257, 0.0018448
6: -0.0015948, -0.0006917, -0.0015245, -0.0007249, -0.0008699, 0.0008328
7: -0.0087858, -0.0073898, -0.0087044, -0.0074575, -0.0013284, 0.0013146
8: 0.0031972, 0.0076150, 0.0034063, 0.0073490, -0.0034675, 0.0035363
9: -0.0047443, -0.0022125, -0.0045966, -0.0023352, -0.0024090, 0.0023840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010099
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010223
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006289, 0.0014182, 0.0006730, 0.0013796, -0.0007414, 0.0007361
1: 0.9928858, 0.9948760, 0.9929829, 0.9947646, -0.0017666, 0.0017797
2: -0.0070932, -0.0045694, -0.0069241, -0.0046738, -0.0024194, 0.0023547
3: 0.0034691, 0.0042945, 0.0035152, 0.0042542, -0.0007085, 0.0007028
4: 0.0023581, 0.0040231, 0.0024094, 0.0038894, -0.0015313, 0.0016137
5: 0.0052322, 0.0071665, 0.0053403, 0.0070721, -0.0018399, 0.0018262
6: -0.0015832, -0.0006866, -0.0015245, -0.0007249, -0.0008583, 0.0008379
7: -0.0087724, -0.0073796, -0.0087044, -0.0074575, -0.0013150, 0.0013248
8: 0.0031656, 0.0075712, 0.0034063, 0.0073490, -0.0035143, 0.0034981
9: -0.0047200, -0.0021940, -0.0045966, -0.0023352, -0.0023847, 0.0024026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010118
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010225
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006305, 0.0014258, 0.0006615, 0.0013609, -0.0007213, 0.0007553
1: 0.9928665, 0.9948716, 0.9930301, 0.9947939, -0.0018161, 0.0017298
2: -0.0071267, -0.0045733, -0.0068421, -0.0046465, -0.0024802, 0.0022687
3: 0.0034708, 0.0043025, 0.0035032, 0.0042347, -0.0006876, 0.0007234
4: 0.0023601, 0.0040496, 0.0023960, 0.0038246, -0.0014645, 0.0016536
5: 0.0052363, 0.0071853, 0.0053121, 0.0070262, -0.0017899, 0.0018732
6: -0.0015948, -0.0006881, -0.0014960, -0.0007149, -0.0008800, 0.0008079
7: -0.0087859, -0.0073826, -0.0086714, -0.0074371, -0.0013488, 0.0012888
8: 0.0031747, 0.0076152, 0.0033434, 0.0072413, -0.0034315, 0.0035748
9: -0.0047444, -0.0021994, -0.0045368, -0.0022983, -0.0024461, 0.0023374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010845, upper bound: 0.0010298
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010845, upper bound: 0.0010298
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006247, 0.0014182, 0.0006615, 0.0013609, -0.0007269, 0.0007478
1: 0.9928857, 0.9948865, 0.9930301, 0.9947939, -0.0017978, 0.0017448
2: -0.0070934, -0.0045595, -0.0068421, -0.0046465, -0.0024469, 0.0022826
3: 0.0034647, 0.0042946, 0.0035032, 0.0042347, -0.0006943, 0.0007159
4: 0.0023533, 0.0040232, 0.0023960, 0.0038246, -0.0014713, 0.0016273
5: 0.0052220, 0.0071666, 0.0053121, 0.0070262, -0.0018042, 0.0018545
6: -0.0015833, -0.0006830, -0.0014960, -0.0007149, -0.0008684, 0.0008130
7: -0.0087725, -0.0073722, -0.0086714, -0.0074371, -0.0013354, 0.0012991
8: 0.0031428, 0.0075714, 0.0033434, 0.0072413, -0.0034794, 0.0035294
9: -0.0047201, -0.0021807, -0.0045368, -0.0022983, -0.0024218, 0.0023561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010845, upper bound: 0.0010325
time: 1.14 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010845, upper bound: 0.0010414
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006347, 0.0014257, 0.0006660, 0.0013728, -0.0007292, 0.0007506
1: 0.9928665, 0.9948614, 0.9930000, 0.9947824, -0.0018038, 0.0017506
2: -0.0071265, -0.0045831, -0.0068944, -0.0046572, -0.0024693, 0.0023113
3: 0.0034751, 0.0043025, 0.0035079, 0.0042471, -0.0006961, 0.0007190
4: 0.0023649, 0.0040494, 0.0024012, 0.0038660, -0.0015011, 0.0016482
5: 0.0052464, 0.0071851, 0.0053231, 0.0070555, -0.0018091, 0.0018620
6: -0.0015948, -0.0006917, -0.0015142, -0.0007188, -0.0008760, 0.0008225
7: -0.0087858, -0.0073898, -0.0086924, -0.0074451, -0.0013407, 0.0013026
8: 0.0031972, 0.0076150, 0.0033680, 0.0073100, -0.0034543, 0.0036149
9: -0.0047443, -0.0022125, -0.0045749, -0.0023128, -0.0024315, 0.0023624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010571, upper bound: 0.0010099
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010571, upper bound: 0.0010233
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006305, 0.0014258, 0.0006532, 0.0013553, -0.0007158, 0.0007636
1: 0.9928665, 0.9948716, 0.9930442, 0.9948147, -0.0018380, 0.0017171
2: -0.0071267, -0.0045733, -0.0068175, -0.0046269, -0.0024998, 0.0022442
3: 0.0034708, 0.0043025, 0.0034945, 0.0042288, -0.0006826, 0.0007330
4: 0.0023601, 0.0040496, 0.0023863, 0.0038052, -0.0014451, 0.0016632
5: 0.0052363, 0.0071853, 0.0052917, 0.0070125, -0.0017762, 0.0018935
6: -0.0015948, -0.0006881, -0.0014875, -0.0007077, -0.0008872, 0.0007994
7: -0.0087859, -0.0073826, -0.0086615, -0.0074224, -0.0013634, 0.0012790
8: 0.0031747, 0.0076152, 0.0032981, 0.0072090, -0.0034232, 0.0036562
9: -0.0047444, -0.0021994, -0.0045188, -0.0022718, -0.0024727, 0.0023194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010298
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010409
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006289, 0.0014182, 0.0006660, 0.0013728, -0.0007349, 0.0007432
1: 0.9928858, 0.9948760, 0.9930000, 0.9947824, -0.0017859, 0.0017660
2: -0.0070932, -0.0045694, -0.0068944, -0.0046572, -0.0024360, 0.0023250
3: 0.0034691, 0.0042945, 0.0035079, 0.0042471, -0.0007028, 0.0007115
4: 0.0023581, 0.0040231, 0.0024012, 0.0038660, -0.0015078, 0.0016219
5: 0.0052322, 0.0071665, 0.0053231, 0.0070555, -0.0018233, 0.0018434
6: -0.0015832, -0.0006866, -0.0015142, -0.0007188, -0.0008644, 0.0008276
7: -0.0087724, -0.0073796, -0.0086924, -0.0074451, -0.0013273, 0.0013129
8: 0.0031656, 0.0075712, 0.0033680, 0.0073100, -0.0035155, 0.0035886
9: -0.0047200, -0.0021940, -0.0045749, -0.0023128, -0.0024072, 0.0023809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010592, upper bound: 0.0010118
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010592, upper bound: 0.0010225
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006247, 0.0014182, 0.0006532, 0.0013553, -0.0007215, 0.0007561
1: 0.9928857, 0.9948865, 0.9930442, 0.9948147, -0.0018195, 0.0017321
2: -0.0070934, -0.0045595, -0.0068175, -0.0046269, -0.0024665, 0.0022580
3: 0.0034647, 0.0042946, 0.0034945, 0.0042288, -0.0006890, 0.0007256
4: 0.0023533, 0.0040232, 0.0023863, 0.0038052, -0.0014519, 0.0016369
5: 0.0052220, 0.0071666, 0.0052917, 0.0070125, -0.0017905, 0.0018749
6: -0.0015833, -0.0006830, -0.0014875, -0.0007077, -0.0008756, 0.0008045
7: -0.0087725, -0.0073722, -0.0086615, -0.0074224, -0.0013500, 0.0012893
8: 0.0031428, 0.0075714, 0.0032981, 0.0072090, -0.0034764, 0.0036351
9: -0.0047201, -0.0021807, -0.0045188, -0.0022718, -0.0024483, 0.0023381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010919, upper bound: 0.0010344
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010919, upper bound: 0.0010432
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006391, 0.0014293, 0.0006561, 0.0013691, -0.0007207, 0.0007641
1: 0.9928576, 0.9948503, 0.9930096, 0.9948073, -0.0018381, 0.0017270
2: -0.0071419, -0.0045935, -0.0068779, -0.0046340, -0.0025080, 0.0022843
3: 0.0034797, 0.0043062, 0.0034976, 0.0042432, -0.0006864, 0.0007326
4: 0.0023700, 0.0040616, 0.0023898, 0.0038529, -0.0014829, 0.0016718
5: 0.0052572, 0.0071938, 0.0052991, 0.0070462, -0.0017890, 0.0018947
6: -0.0016001, -0.0006955, -0.0015085, -0.0007103, -0.0008898, 0.0008130
7: -0.0087920, -0.0073976, -0.0086858, -0.0074277, -0.0013643, 0.0012882
8: 0.0032212, 0.0076352, 0.0033144, 0.0072883, -0.0034166, 0.0037382
9: -0.0047555, -0.0022267, -0.0045629, -0.0022813, -0.0024742, 0.0023362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010834, upper bound: 0.0010100
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010834, upper bound: 0.0010121
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006349, 0.0014293, 0.0006440, 0.0013500, -0.0007059, 0.0007764
1: 0.9928576, 0.9948608, 0.9930576, 0.9948378, -0.0018697, 0.0016912
2: -0.0071421, -0.0045837, -0.0067942, -0.0046051, -0.0025370, 0.0022105
3: 0.0034754, 0.0043062, 0.0034849, 0.0042232, -0.0006719, 0.0007456
4: 0.0023652, 0.0040617, 0.0023757, 0.0037868, -0.0014216, 0.0016861
5: 0.0052471, 0.0071939, 0.0052692, 0.0069995, -0.0017524, 0.0019246
6: -0.0016002, -0.0006919, -0.0014794, -0.0006997, -0.0009005, 0.0007875
7: -0.0087921, -0.0073903, -0.0086521, -0.0074062, -0.0013858, 0.0012618
8: 0.0031986, 0.0076355, 0.0032480, 0.0071784, -0.0033823, 0.0037700
9: -0.0047557, -0.0022134, -0.0045018, -0.0022424, -0.0025133, 0.0022884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010996, upper bound: 0.0010299
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010996, upper bound: 0.0010330
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006219, 0.0014182, 0.0006561, 0.0013691, -0.0007378, 0.0007529
1: 0.9928856, 0.9948934, 0.9930096, 0.9948073, -0.0018076, 0.0017681
2: -0.0070936, -0.0045529, -0.0068779, -0.0046340, -0.0024596, 0.0023250
3: 0.0034618, 0.0042946, 0.0034976, 0.0042432, -0.0007026, 0.0007191
4: 0.0023501, 0.0040234, 0.0023898, 0.0038529, -0.0015028, 0.0016336
5: 0.0052152, 0.0071667, 0.0052991, 0.0070462, -0.0018311, 0.0018677
6: -0.0015833, -0.0006806, -0.0015085, -0.0007103, -0.0008731, 0.0008278
7: -0.0087726, -0.0073673, -0.0086858, -0.0074277, -0.0013448, 0.0013185
8: 0.0031276, 0.0075717, 0.0033144, 0.0072883, -0.0034728, 0.0036078
9: -0.0047202, -0.0021718, -0.0045629, -0.0022813, -0.0024389, 0.0023911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011029, upper bound: 0.0010368
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011029, upper bound: 0.0010370
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006177, 0.0014183, 0.0006440, 0.0013500, -0.0007231, 0.0007652
1: 0.9928854, 0.9949040, 0.9930576, 0.9948378, -0.0018394, 0.0017325
2: -0.0070937, -0.0045430, -0.0067942, -0.0046051, -0.0024886, 0.0022512
3: 0.0034574, 0.0042947, 0.0034849, 0.0042232, -0.0006881, 0.0007324
4: 0.0023452, 0.0040235, 0.0023757, 0.0037868, -0.0014416, 0.0016478
5: 0.0052049, 0.0071669, 0.0052692, 0.0069995, -0.0017946, 0.0018976
6: -0.0015834, -0.0006770, -0.0014794, -0.0006997, -0.0008837, 0.0008024
7: -0.0087726, -0.0073599, -0.0086521, -0.0074062, -0.0013664, 0.0012922
8: 0.0031048, 0.0075719, 0.0032480, 0.0071784, -0.0034353, 0.0036383
9: -0.0047204, -0.0021584, -0.0045018, -0.0022424, -0.0024780, 0.0023435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011135, upper bound: 0.0010541
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011135, upper bound: 0.0010553
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0006551, 0.0014364, 0.0006262, 0.0013434, -0.0006793, 0.0008013
1: 0.9928396, 0.9948098, 0.9930742, 0.9948826, -0.0019336, 0.0016251
2: -0.0071734, -0.0046315, -0.0067652, -0.0045631, -0.0026103, 0.0021337
3: 0.0034965, 0.0043137, 0.0034663, 0.0042163, -0.0006448, 0.0007732
4: 0.0023886, 0.0040865, 0.0023551, 0.0037638, -0.0013752, 0.0017314
5: 0.0052965, 0.0072114, 0.0052257, 0.0069833, -0.0016868, 0.0019856
6: -0.0016110, -0.0007094, -0.0014693, -0.0006844, -0.0009267, 0.0007600
7: -0.0088047, -0.0074259, -0.0086404, -0.0073749, -0.0014298, 0.0012146
8: 0.0033087, 0.0076766, 0.0031512, 0.0071402, -0.0032394, 0.0039745
9: -0.0047785, -0.0022780, -0.0044806, -0.0021856, -0.0025929, 0.0022026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010571, upper bound: 0.0010100
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010299
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006262, 0.0013434, -0.0006968, 0.0007903
1: 0.9928666, 0.9948542, 0.9930742, 0.9948826, -0.0019033, 0.0016678
2: -0.0071265, -0.0045897, -0.0067652, -0.0045631, -0.0025634, 0.0021755
3: 0.0034781, 0.0043025, 0.0034663, 0.0042163, -0.0006614, 0.0007599
4: 0.0023681, 0.0040494, 0.0023551, 0.0037638, -0.0013957, 0.0016943
5: 0.0052533, 0.0071851, 0.0052257, 0.0069833, -0.0017300, 0.0019594
6: -0.0015947, -0.0006941, -0.0014693, -0.0006844, -0.0009104, 0.0007753
7: -0.0087858, -0.0073948, -0.0086404, -0.0073749, -0.0014109, 0.0012457
8: 0.0032124, 0.0076149, 0.0031512, 0.0071402, -0.0032915, 0.0038469
9: -0.0047442, -0.0022215, -0.0044806, -0.0021856, -0.0025587, 0.0022591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010571, upper bound: 0.0010377
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010547
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0006489, 0.0014292, 0.0006262, 0.0013434, -0.0006855, 0.0007942
1: 0.9928579, 0.9948256, 0.9930742, 0.9948826, -0.0019166, 0.0016420
2: -0.0071416, -0.0046168, -0.0067652, -0.0045631, -0.0025785, 0.0021483
3: 0.0034900, 0.0043061, 0.0034663, 0.0042163, -0.0006523, 0.0007660
4: 0.0023814, 0.0040613, 0.0023551, 0.0037638, -0.0013824, 0.0017063
5: 0.0052813, 0.0071936, 0.0052257, 0.0069833, -0.0017019, 0.0019678
6: -0.0016000, -0.0007040, -0.0014693, -0.0006844, -0.0009156, 0.0007653
7: -0.0087919, -0.0074150, -0.0086404, -0.0073749, -0.0014169, 0.0012255
8: 0.0032749, 0.0076348, 0.0031512, 0.0071402, -0.0032907, 0.0039440
9: -0.0047553, -0.0022582, -0.0044806, -0.0021856, -0.0025697, 0.0022225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010592, upper bound: 0.0010121
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010919, upper bound: 0.0010346
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006262, 0.0013434, -0.0007025, 0.0007829
1: 0.9928859, 0.9948689, 0.9930742, 0.9948826, -0.0018855, 0.0016836
2: -0.0070932, -0.0045761, -0.0067652, -0.0045631, -0.0025300, 0.0021890
3: 0.0034720, 0.0042945, 0.0034663, 0.0042163, -0.0006686, 0.0007526
4: 0.0023615, 0.0040230, 0.0023551, 0.0037638, -0.0014024, 0.0016680
5: 0.0052392, 0.0071665, 0.0052257, 0.0069833, -0.0017440, 0.0019408
6: -0.0015832, -0.0006891, -0.0014693, -0.0006844, -0.0008988, 0.0007802
7: -0.0087724, -0.0073846, -0.0086404, -0.0073749, -0.0013975, 0.0012558
8: 0.0031812, 0.0075711, 0.0031512, 0.0071402, -0.0033426, 0.0038098
9: -0.0047199, -0.0022032, -0.0044806, -0.0021856, -0.0025344, 0.0022774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010592, upper bound: 0.0010118
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010919, upper bound: 0.0010561
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0014364, 0.0006531, 0.0014447, -0.0007759, 0.0007737
1: 0.9928398, 0.9947996, 0.9928188, 0.9948150, -0.0018559, 0.0018618
2: -0.0071732, -0.0046411, -0.0072096, -0.0046267, -0.0025465, 0.0025685
3: 0.0035008, 0.0043136, 0.0034944, 0.0043223, -0.0007401, 0.0007374
4: 0.0023933, 0.0040863, 0.0023863, 0.0041151, -0.0017218, 0.0017000
5: 0.0053064, 0.0072112, 0.0052916, 0.0072316, -0.0019251, 0.0019197
6: -0.0016110, -0.0007129, -0.0016236, -0.0007076, -0.0009033, 0.0009107
7: -0.0088046, -0.0074330, -0.0088192, -0.0074223, -0.0013823, 0.0013862
8: 0.0033308, 0.0076763, 0.0032977, 0.0077242, -0.0035668, 0.0035616
9: -0.0047783, -0.0022910, -0.0048049, -0.0022716, -0.0025068, 0.0025139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010425
time: 0.89 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010430
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006530, 0.0014291, 0.0006531, 0.0014447, -0.0007820, 0.0007665
1: 0.9928580, 0.9948150, 0.9928188, 0.9948150, -0.0018382, 0.0018778
2: -0.0071414, -0.0046266, -0.0072096, -0.0046267, -0.0025147, 0.0025830
3: 0.0034943, 0.0043060, 0.0034944, 0.0043223, -0.0007473, 0.0007303
4: 0.0023862, 0.0040612, 0.0023863, 0.0041151, -0.0017289, 0.0016749
5: 0.0052914, 0.0071935, 0.0052916, 0.0072316, -0.0019401, 0.0019019
6: -0.0015999, -0.0007076, -0.0016236, -0.0007076, -0.0008923, 0.0009160
7: -0.0087918, -0.0074222, -0.0088192, -0.0074223, -0.0013695, 0.0013970
8: 0.0032974, 0.0076345, 0.0032977, 0.0077242, -0.0036393, 0.0035330
9: -0.0047551, -0.0022714, -0.0048049, -0.0022716, -0.0024836, 0.0025335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010469
time: 0.95 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010477
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006551, 0.0014364, 0.0006414, 0.0014257, -0.0007610, 0.0007855
1: 0.9928396, 0.9948098, 0.9928667, 0.9948444, -0.0018859, 0.0018242
2: -0.0071734, -0.0046315, -0.0071262, -0.0045991, -0.0025743, 0.0024947
3: 0.0034965, 0.0043137, 0.0034822, 0.0043024, -0.0007249, 0.0007499
4: 0.0023886, 0.0040865, 0.0023727, 0.0040492, -0.0016606, 0.0017138
5: 0.0052965, 0.0072114, 0.0052630, 0.0071850, -0.0018885, 0.0019484
6: -0.0016110, -0.0007094, -0.0015947, -0.0006975, -0.0009135, 0.0008853
7: -0.0088047, -0.0074259, -0.0087857, -0.0074017, -0.0014029, 0.0013598
8: 0.0033087, 0.0076766, 0.0032340, 0.0076146, -0.0035346, 0.0035948
9: -0.0047785, -0.0022780, -0.0047441, -0.0022342, -0.0025443, 0.0024661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010899, upper bound: 0.0010569
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010899, upper bound: 0.0010570
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006489, 0.0014292, 0.0006414, 0.0014257, -0.0007671, 0.0007782
1: 0.9928579, 0.9948256, 0.9928667, 0.9948444, -0.0018681, 0.0018405
2: -0.0071416, -0.0046168, -0.0071262, -0.0045991, -0.0025425, 0.0025094
3: 0.0034900, 0.0043061, 0.0034822, 0.0043024, -0.0007322, 0.0007426
4: 0.0023814, 0.0040613, 0.0023727, 0.0040492, -0.0016678, 0.0016886
5: 0.0052813, 0.0071936, 0.0052630, 0.0071850, -0.0019036, 0.0019306
6: -0.0016000, -0.0007040, -0.0015947, -0.0006975, -0.0009025, 0.0008907
7: -0.0087919, -0.0074150, -0.0087857, -0.0074017, -0.0013901, 0.0013707
8: 0.0032749, 0.0076348, 0.0032340, 0.0076146, -0.0036070, 0.0035659
9: -0.0047553, -0.0022582, -0.0047441, -0.0022342, -0.0025211, 0.0024859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010899, upper bound: 0.0010618
time: 1.08 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010899, upper bound: 0.0010570
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006410, 0.0014370, 0.0006592, 0.0014364, -0.0007858, 0.0007682
1: 0.9928381, 0.9948453, 0.9928398, 0.9947996, -0.0018431, 0.0018876
2: -0.0071759, -0.0045981, -0.0071732, -0.0046411, -0.0025349, 0.0025751
3: 0.0034818, 0.0043143, 0.0035008, 0.0043136, -0.0007509, 0.0007325
4: 0.0023722, 0.0040885, 0.0023933, 0.0040863, -0.0017141, 0.0016952
5: 0.0052620, 0.0072128, 0.0053064, 0.0072112, -0.0019493, 0.0019063
6: -0.0016119, -0.0006972, -0.0016110, -0.0007129, -0.0008990, 0.0009138
7: -0.0088057, -0.0074010, -0.0088046, -0.0074330, -0.0013727, 0.0014036
8: 0.0032319, 0.0076800, 0.0033308, 0.0076763, -0.0036884, 0.0035757
9: -0.0047804, -0.0022329, -0.0047783, -0.0022910, -0.0024894, 0.0025454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010549
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010559
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006287, 0.0014182, 0.0006551, 0.0014364, -0.0007983, 0.0007535
1: 0.9928857, 0.9948764, 0.9928396, 0.9948098, -0.0018065, 0.0019189
2: -0.0070933, -0.0045689, -0.0071734, -0.0046315, -0.0024618, 0.0026045
3: 0.0034689, 0.0042946, 0.0034965, 0.0043137, -0.0007640, 0.0007176
4: 0.0023579, 0.0040231, 0.0023886, 0.0040865, -0.0017286, 0.0016345
5: 0.0052317, 0.0071666, 0.0052965, 0.0072114, -0.0019796, 0.0018701
6: -0.0015832, -0.0006865, -0.0016110, -0.0007094, -0.0008739, 0.0009246
7: -0.0087724, -0.0073792, -0.0088047, -0.0074259, -0.0013466, 0.0014254
8: 0.0031645, 0.0075713, 0.0033087, 0.0076766, -0.0037219, 0.0035516
9: -0.0047200, -0.0021934, -0.0047785, -0.0022780, -0.0024420, 0.0025851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010902, upper bound: 0.0010664
time: 0.92 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010902, upper bound: 0.0010684
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006375, 0.0014257, -0.0007785, 0.0007785
1: 0.9928666, 0.9948542, 0.9928666, 0.9948542, -0.0018667, 0.0018667
2: -0.0071265, -0.0045897, -0.0071265, -0.0045897, -0.0025368, 0.0025368
3: 0.0034781, 0.0043025, 0.0034781, 0.0043025, -0.0007415, 0.0007415
4: 0.0023681, 0.0040494, 0.0023681, 0.0040494, -0.0016813, 0.0016813
5: 0.0052533, 0.0071851, 0.0052533, 0.0071851, -0.0019319, 0.0019319
6: -0.0015947, -0.0006941, -0.0015947, -0.0006941, -0.0009007, 0.0009007
7: -0.0087858, -0.0073948, -0.0087858, -0.0073948, -0.0013910, 0.0013910
8: 0.0032124, 0.0076149, 0.0032124, 0.0076149, -0.0035941, 0.0035941
9: -0.0047442, -0.0022215, -0.0047442, -0.0022215, -0.0025227, 0.0025227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011015, upper bound: 0.0010691
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011202, upper bound: 0.0010782
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006375, 0.0014257, -0.0007842, 0.0007710
1: 0.9928859, 0.9948689, 0.9928666, 0.9948542, -0.0018490, 0.0018817
2: -0.0070932, -0.0045761, -0.0071265, -0.0045897, -0.0025034, 0.0025503
3: 0.0034720, 0.0042945, 0.0034781, 0.0043025, -0.0007484, 0.0007343
4: 0.0023615, 0.0040230, 0.0023681, 0.0040494, -0.0016879, 0.0016549
5: 0.0052392, 0.0071665, 0.0052533, 0.0071851, -0.0019459, 0.0019133
6: -0.0015832, -0.0006891, -0.0015947, -0.0006941, -0.0008891, 0.0009056
7: -0.0087724, -0.0073846, -0.0087858, -0.0073948, -0.0013776, 0.0014011
8: 0.0031812, 0.0075711, 0.0032124, 0.0076149, -0.0036654, 0.0035656
9: -0.0047199, -0.0022032, -0.0047442, -0.0022215, -0.0024984, 0.0025411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011015, upper bound: 0.0010704
time: 1.01 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011202, upper bound: 0.0010794
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0014364, 0.0006480, 0.0014369, -0.0007681, 0.0007786
1: 0.9928398, 0.9947996, 0.9928385, 0.9948276, -0.0018689, 0.0018429
2: -0.0071732, -0.0046411, -0.0071756, -0.0046147, -0.0025585, 0.0025345
3: 0.0035008, 0.0043136, 0.0034891, 0.0043142, -0.0007324, 0.0007435
4: 0.0023933, 0.0040863, 0.0023804, 0.0040882, -0.0016949, 0.0017059
5: 0.0053064, 0.0072112, 0.0052791, 0.0072126, -0.0019061, 0.0019321
6: -0.0016110, -0.0007129, -0.0016118, -0.0007032, -0.0009077, 0.0008989
7: -0.0088046, -0.0074330, -0.0088055, -0.0074134, -0.0013912, 0.0013725
8: 0.0033308, 0.0076763, 0.0032700, 0.0076794, -0.0035300, 0.0036288
9: -0.0047783, -0.0022910, -0.0047801, -0.0022553, -0.0025230, 0.0024891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010663, upper bound: 0.0010426
time: 1.00 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010663, upper bound: 0.0010430
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006551, 0.0014364, 0.0006355, 0.0014181, -0.0007534, 0.0007913
1: 0.9928396, 0.9948098, 0.9928859, 0.9948592, -0.0019012, 0.0018063
2: -0.0071734, -0.0046315, -0.0070929, -0.0045851, -0.0025883, 0.0024614
3: 0.0034965, 0.0043137, 0.0034760, 0.0042945, -0.0007175, 0.0007568
4: 0.0023886, 0.0040865, 0.0023659, 0.0040229, -0.0016343, 0.0017206
5: 0.0052965, 0.0072114, 0.0052485, 0.0071664, -0.0018699, 0.0019628
6: -0.0016110, -0.0007094, -0.0015831, -0.0006924, -0.0009186, 0.0008737
7: -0.0088047, -0.0074259, -0.0087723, -0.0073914, -0.0014133, 0.0013464
8: 0.0033087, 0.0076766, 0.0032019, 0.0075708, -0.0035033, 0.0036640
9: -0.0047785, -0.0022780, -0.0047198, -0.0022154, -0.0025631, 0.0024418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010923, upper bound: 0.0010570
time: 0.95 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010923, upper bound: 0.0010571
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006530, 0.0014291, 0.0006480, 0.0014369, -0.0007743, 0.0007715
1: 0.9928580, 0.9948150, 0.9928385, 0.9948276, -0.0018531, 0.0018603
2: -0.0071414, -0.0046266, -0.0071756, -0.0046147, -0.0025267, 0.0025490
3: 0.0034943, 0.0043060, 0.0034891, 0.0043142, -0.0007404, 0.0007373
4: 0.0023862, 0.0040612, 0.0023804, 0.0040882, -0.0017020, 0.0016808
5: 0.0052914, 0.0071935, 0.0052791, 0.0072126, -0.0019211, 0.0019143
6: -0.0015999, -0.0007076, -0.0016118, -0.0007032, -0.0008967, 0.0009042
7: -0.0087918, -0.0074222, -0.0088055, -0.0074134, -0.0013784, 0.0013833
8: 0.0032974, 0.0076345, 0.0032700, 0.0076794, -0.0036202, 0.0036124
9: -0.0047551, -0.0022714, -0.0047801, -0.0022553, -0.0024998, 0.0025087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010700, upper bound: 0.0010502
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010700, upper bound: 0.0010502
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006489, 0.0014292, 0.0006355, 0.0014181, -0.0007597, 0.0007841
1: 0.9928579, 0.9948256, 0.9928859, 0.9948592, -0.0018850, 0.0018235
2: -0.0071416, -0.0046168, -0.0070929, -0.0045851, -0.0025564, 0.0024761
3: 0.0034900, 0.0043061, 0.0034760, 0.0042945, -0.0007255, 0.0007504
4: 0.0023814, 0.0040613, 0.0023659, 0.0040229, -0.0016414, 0.0016955
5: 0.0052813, 0.0071936, 0.0052485, 0.0071664, -0.0018850, 0.0019450
6: -0.0016000, -0.0007040, -0.0015831, -0.0006924, -0.0009076, 0.0008791
7: -0.0087919, -0.0074150, -0.0087723, -0.0073914, -0.0014005, 0.0013573
8: 0.0032749, 0.0076348, 0.0032019, 0.0075708, -0.0035814, 0.0036472
9: -0.0047553, -0.0022582, -0.0047198, -0.0022154, -0.0025399, 0.0024616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011022, upper bound: 0.0010681
time: 1.06 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011022, upper bound: 0.0010682
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006489, 0.0014292, -0.0007823, 0.0007672
1: 0.9928666, 0.9948542, 0.9928579, 0.9948256, -0.0018407, 0.0018791
2: -0.0071265, -0.0045897, -0.0071416, -0.0046168, -0.0025097, 0.0025519
3: 0.0034781, 0.0043025, 0.0034900, 0.0043061, -0.0007475, 0.0007323
4: 0.0023681, 0.0040494, 0.0023814, 0.0040613, -0.0016932, 0.0016680
5: 0.0052533, 0.0071851, 0.0052813, 0.0071936, -0.0019403, 0.0019038
6: -0.0015947, -0.0006941, -0.0016000, -0.0007040, -0.0008907, 0.0009059
7: -0.0087858, -0.0073948, -0.0087919, -0.0074150, -0.0013708, 0.0013971
8: 0.0032124, 0.0076149, 0.0032749, 0.0076348, -0.0036920, 0.0036077
9: -0.0047442, -0.0022215, -0.0047553, -0.0022582, -0.0024861, 0.0025337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010685, upper bound: 0.0010558
time: 0.97 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010926, upper bound: 0.0010668
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006317, 0.0014181, -0.0007710, 0.0007842
1: 0.9928666, 0.9948542, 0.9928859, 0.9948689, -0.0018817, 0.0018490
2: -0.0071265, -0.0045897, -0.0070932, -0.0045761, -0.0025503, 0.0025034
3: 0.0034781, 0.0043025, 0.0034720, 0.0042945, -0.0007343, 0.0007484
4: 0.0023681, 0.0040494, 0.0023615, 0.0040230, -0.0016549, 0.0016879
5: 0.0052533, 0.0071851, 0.0052392, 0.0071665, -0.0019133, 0.0019459
6: -0.0015947, -0.0006941, -0.0015832, -0.0006891, -0.0009056, 0.0008891
7: -0.0087858, -0.0073948, -0.0087724, -0.0073846, -0.0014011, 0.0013776
8: 0.0032124, 0.0076149, 0.0031812, 0.0075711, -0.0035656, 0.0036654
9: -0.0047442, -0.0022215, -0.0047199, -0.0022032, -0.0025411, 0.0024984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010679, upper bound: 0.0010680
time: 1.23 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010926, upper bound: 0.0010788
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006489, 0.0014292, -0.0007880, 0.0007597
1: 0.9928859, 0.9948689, 0.9928579, 0.9948256, -0.0018237, 0.0018960
2: -0.0070932, -0.0045761, -0.0071416, -0.0046168, -0.0024763, 0.0025654
3: 0.0034720, 0.0042945, 0.0034900, 0.0043061, -0.0007554, 0.0007255
4: 0.0023615, 0.0040230, 0.0023814, 0.0040613, -0.0016999, 0.0016416
5: 0.0052392, 0.0071665, 0.0052813, 0.0071936, -0.0019543, 0.0018852
6: -0.0015832, -0.0006891, -0.0016000, -0.0007040, -0.0008792, 0.0009109
7: -0.0087724, -0.0073846, -0.0087919, -0.0074150, -0.0013574, 0.0014072
8: 0.0031812, 0.0075711, 0.0032749, 0.0076348, -0.0037692, 0.0035821
9: -0.0047199, -0.0022032, -0.0047553, -0.0022582, -0.0024618, 0.0025521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010725, upper bound: 0.0010602
time: 1.04 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011028, upper bound: 0.0010748
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006317, 0.0014181, -0.0007768, 0.0007768
1: 0.9928859, 0.9948689, 0.9928859, 0.9948689, -0.0018651, 0.0018651
2: -0.0070932, -0.0045761, -0.0070932, -0.0045761, -0.0025170, 0.0025170
3: 0.0034720, 0.0042945, 0.0034720, 0.0042945, -0.0007418, 0.0007418
4: 0.0023615, 0.0040230, 0.0023615, 0.0040230, -0.0016616, 0.0016616
5: 0.0052392, 0.0071665, 0.0052392, 0.0071665, -0.0019273, 0.0019273
6: -0.0015832, -0.0006891, -0.0015832, -0.0006891, -0.0008941, 0.0008941
7: -0.0087724, -0.0073846, -0.0087724, -0.0073846, -0.0013877, 0.0013877
8: 0.0031812, 0.0075711, 0.0031812, 0.0075711, -0.0036355, 0.0036355
9: -0.0047199, -0.0022032, -0.0047199, -0.0022032, -0.0025167, 0.0025167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010715, upper bound: 0.0010713
time: 0.97 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011028, upper bound: 0.0010846
time: 1.05 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.40 seconds
IS_A1_B1_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010556, upper bound: 0.0010556
IS_A1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010556, upper bound: 0.0010561
IS_A1_B1_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010234
IS_A1_B1_A2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010408
IS_A1_B1_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010234
IS_A1_B1_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010432
IS_A1_B1_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0010368
IS_A1_B1_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010548, upper bound: 0.0010556
IS_A1_B1_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010350, upper bound: 0.0010377
IS_A1_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010548, upper bound: 0.0010570
IS_A1_B2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010597
IS_A1_B2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010612
IS_A1_B2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010605
IS_A1_B2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010619
IS_A1_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010845
IS_A1_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010851
IS_A1_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010869
IS_A1_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010874
IS_A1_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010571
IS_A1_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010099, upper bound: 0.0010586
IS_A1_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010848
IS_A1_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010298, upper bound: 0.0010848
IS_A1_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010106, upper bound: 0.0010574
IS_A1_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010106, upper bound: 0.0010591
IS_A1_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010341, upper bound: 0.0010913
IS_A1_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010341, upper bound: 0.0010916
IS_A1_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010834
IS_A1_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010845
IS_A1_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010996
IS_A1_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0011006
IS_A1_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010350, upper bound: 0.0011040
IS_A1_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010350, upper bound: 0.0011040
IS_A1_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010535, upper bound: 0.0011154
IS_A1_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010535, upper bound: 0.0011154
IS_A1_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010787
IS_A1_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0010975
IS_A1_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010100, upper bound: 0.0010977
IS_A1_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010299, upper bound: 0.0011105
IS_A1_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010107, upper bound: 0.0010792
IS_A1_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010343, upper bound: 0.0011021
IS_A1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010107, upper bound: 0.0010978
IS_A1_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010343, upper bound: 0.0011021
IS_A2_B1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010099
IS_A2_B1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010223
IS_A2_B1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010118
IS_A2_B1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010597, upper bound: 0.0010225
IS_A2_B1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010845, upper bound: 0.0010298
IS_A2_B1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010845, upper bound: 0.0010298
IS_A2_B1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010845, upper bound: 0.0010325
IS_A2_B1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010845, upper bound: 0.0010414
IS_A2_B1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010571, upper bound: 0.0010099
IS_A2_B1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010571, upper bound: 0.0010233
IS_A2_B1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010298
IS_A2_B1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010409
IS_A2_B1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010592, upper bound: 0.0010118
IS_A2_B1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010592, upper bound: 0.0010225
IS_A2_B1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010919, upper bound: 0.0010344
IS_A2_B1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010919, upper bound: 0.0010432
IS_A2_B1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010834, upper bound: 0.0010100
IS_A2_B1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010834, upper bound: 0.0010121
IS_A2_B1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010996, upper bound: 0.0010299
IS_A2_B1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010996, upper bound: 0.0010330
IS_A2_B1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011029, upper bound: 0.0010368
IS_A2_B1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011029, upper bound: 0.0010370
IS_A2_B1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011135, upper bound: 0.0010541
IS_A2_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011135, upper bound: 0.0010553
IS_A2_B1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010571, upper bound: 0.0010100
IS_A2_B1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010299
IS_A2_B1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010571, upper bound: 0.0010377
IS_A2_B1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010848, upper bound: 0.0010547
IS_A2_B1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010592, upper bound: 0.0010121
IS_A2_B1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010919, upper bound: 0.0010346
IS_A2_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010592, upper bound: 0.0010118
IS_A2_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010919, upper bound: 0.0010561
IS_A2_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010425
IS_A2_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010430
IS_A2_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010469
IS_A2_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010477
IS_A2_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010899, upper bound: 0.0010569
IS_A2_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010899, upper bound: 0.0010570
IS_A2_B2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010899, upper bound: 0.0010618
IS_A2_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010899, upper bound: 0.0010570
IS_A2_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010549
IS_A2_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010656, upper bound: 0.0010559
IS_A2_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010902, upper bound: 0.0010664
IS_A2_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010902, upper bound: 0.0010684
IS_A2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011015, upper bound: 0.0010691
IS_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011202, upper bound: 0.0010782
IS_A2_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011015, upper bound: 0.0010704
IS_A2_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011202, upper bound: 0.0010794
IS_A2_B2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010663, upper bound: 0.0010426
IS_A2_B2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010663, upper bound: 0.0010430
IS_A2_B2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010923, upper bound: 0.0010570
IS_A2_B2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010923, upper bound: 0.0010571
IS_A2_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010700, upper bound: 0.0010502
IS_A2_B2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010700, upper bound: 0.0010502
IS_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011022, upper bound: 0.0010681
IS_A2_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011022, upper bound: 0.0010682
IS_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010685, upper bound: 0.0010558
IS_A2_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010926, upper bound: 0.0010668
IS_A2_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010679, upper bound: 0.0010680
IS_A2_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010926, upper bound: 0.0010788
IS_A2_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010725, upper bound: 0.0010602
IS_A2_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011028, upper bound: 0.0010748
IS_A2_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0010715, upper bound: 0.0010713
IS_A2_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 1, lower bound: -0.0011028, upper bound: 0.0010846

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0006368, 0.0013433, 0.0006333, 0.0013500, -0.0007038, 0.0007008
1: 0.9930744, 0.9948560, 0.9930574, 0.9948649, -0.0016759, 0.0016827
2: -0.0067649, -0.0045882, -0.0067945, -0.0045798, -0.0021851, 0.0022063
3: 0.0034774, 0.0042163, 0.0034737, 0.0042233, -0.0006670, 0.0006641
4: 0.0023674, 0.0037636, 0.0023632, 0.0037870, -0.0014196, 0.0014004
5: 0.0052517, 0.0069831, 0.0052430, 0.0069996, -0.0017480, 0.0017401
6: -0.0014693, -0.0006935, -0.0014795, -0.0006905, -0.0007788, 0.0007860
7: -0.0086403, -0.0073936, -0.0086522, -0.0073874, -0.0012530, 0.0012586
8: 0.0032089, 0.0071399, 0.0031896, 0.0071787, -0.0032533, 0.0032845
9: -0.0044804, -0.0022194, -0.0045020, -0.0022081, -0.0022723, 0.0022826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010399, upper bound: 0.0010370
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010399, upper bound: 0.0010557
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0006368, 0.0013433, 0.0006262, 0.0013434, -0.0006973, 0.0007078
1: 0.9930744, 0.9948560, 0.9930742, 0.9948826, -0.0016941, 0.0016667
2: -0.0067649, -0.0045882, -0.0067652, -0.0045631, -0.0022018, 0.0021770
3: 0.0034774, 0.0042163, 0.0034663, 0.0042163, -0.0006603, 0.0006722
4: 0.0023674, 0.0037636, 0.0023551, 0.0037638, -0.0013965, 0.0014086
5: 0.0052517, 0.0069831, 0.0052257, 0.0069833, -0.0017316, 0.0017574
6: -0.0014693, -0.0006935, -0.0014693, -0.0006844, -0.0007849, 0.0007758
7: -0.0086403, -0.0073936, -0.0086404, -0.0073749, -0.0012654, 0.0012468
8: 0.0032089, 0.0071399, 0.0031512, 0.0071402, -0.0032115, 0.0033266
9: -0.0044804, -0.0022194, -0.0044806, -0.0021856, -0.0022949, 0.0022612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0010370
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0010570
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0006730, 0.0013796, 0.0006592, 0.0014364, -0.0007543, 0.0007111
1: 0.9929829, 0.9947646, 0.9928398, 0.9947996, -0.0017022, 0.0018115
2: -0.0069241, -0.0046738, -0.0071732, -0.0046411, -0.0022830, 0.0024994
3: 0.0035152, 0.0042542, 0.0035008, 0.0043136, -0.0007209, 0.0006759
4: 0.0024094, 0.0038894, 0.0023933, 0.0040863, -0.0016769, 0.0014961
5: 0.0053403, 0.0070721, 0.0053064, 0.0072112, -0.0018709, 0.0017656
6: -0.0015245, -0.0007249, -0.0016110, -0.0007129, -0.0008116, 0.0008861
7: -0.0087044, -0.0074575, -0.0088046, -0.0074330, -0.0012713, 0.0013471
8: 0.0034063, 0.0073490, 0.0033308, 0.0076763, -0.0035903, 0.0033258
9: -0.0045966, -0.0023352, -0.0047783, -0.0022910, -0.0023056, 0.0024431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010039, upper bound: 0.0010517
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010039, upper bound: 0.0010597
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0006730, 0.0013796, 0.0006415, 0.0014257, -0.0007436, 0.0007289
1: 0.9929829, 0.9947646, 0.9928668, 0.9948441, -0.0017480, 0.0017846
2: -0.0069241, -0.0046738, -0.0071263, -0.0045994, -0.0023247, 0.0024524
3: 0.0035152, 0.0042542, 0.0034823, 0.0043024, -0.0007101, 0.0006949
4: 0.0024094, 0.0038894, 0.0023729, 0.0040492, -0.0016399, 0.0015166
5: 0.0053403, 0.0070721, 0.0052633, 0.0071850, -0.0018447, 0.0018088
6: -0.0015245, -0.0007249, -0.0015947, -0.0006976, -0.0008269, 0.0008698
7: -0.0087044, -0.0074575, -0.0087857, -0.0074020, -0.0013024, 0.0013283
8: 0.0034063, 0.0073490, 0.0032347, 0.0076147, -0.0035355, 0.0034521
9: -0.0045966, -0.0023352, -0.0047441, -0.0022346, -0.0023620, 0.0024089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010039, upper bound: 0.0010544
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010039, upper bound: 0.0010612
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0006730, 0.0013796, 0.0006530, 0.0014291, -0.0007470, 0.0007172
1: 0.9929829, 0.9947646, 0.9928580, 0.9948150, -0.0017179, 0.0017936
2: -0.0069241, -0.0046738, -0.0071414, -0.0046266, -0.0022975, 0.0024675
3: 0.0035152, 0.0042542, 0.0034943, 0.0043060, -0.0007135, 0.0006828
4: 0.0024094, 0.0038894, 0.0023862, 0.0040612, -0.0016518, 0.0015032
5: 0.0053403, 0.0070721, 0.0052914, 0.0071935, -0.0018531, 0.0017806
6: -0.0015245, -0.0007249, -0.0015999, -0.0007076, -0.0008169, 0.0008751
7: -0.0087044, -0.0074575, -0.0087918, -0.0074222, -0.0012821, 0.0013343
8: 0.0034063, 0.0073490, 0.0032974, 0.0076345, -0.0035541, 0.0033728
9: -0.0045966, -0.0023352, -0.0047551, -0.0022714, -0.0023252, 0.0024199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010523
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010523
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0006730, 0.0013796, 0.0006359, 0.0014181, -0.0007360, 0.0007345
1: 0.9929829, 0.9947646, 0.9928859, 0.9948581, -0.0017626, 0.0017665
2: -0.0069241, -0.0046738, -0.0070929, -0.0045860, -0.0023381, 0.0024191
3: 0.0035152, 0.0042542, 0.0034764, 0.0042945, -0.0007027, 0.0007017
4: 0.0024094, 0.0038894, 0.0023663, 0.0040229, -0.0016135, 0.0015231
5: 0.0053403, 0.0070721, 0.0052494, 0.0071664, -0.0018261, 0.0018226
6: -0.0015245, -0.0007249, -0.0015831, -0.0006927, -0.0008318, 0.0008583
7: -0.0087044, -0.0074575, -0.0087723, -0.0073920, -0.0013124, 0.0013149
8: 0.0034063, 0.0073490, 0.0032039, 0.0075709, -0.0034973, 0.0034932
9: -0.0045966, -0.0023352, -0.0047198, -0.0022165, -0.0023801, 0.0023846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010547
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010619
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0006615, 0.0013609, 0.0006551, 0.0014364, -0.0007660, 0.0006966
1: 0.9930301, 0.9947939, 0.9928396, 0.9948098, -0.0016671, 0.0018423
2: -0.0068421, -0.0046465, -0.0071734, -0.0046315, -0.0022106, 0.0025269
3: 0.0035032, 0.0042347, 0.0034965, 0.0043137, -0.0007341, 0.0006617
4: 0.0023960, 0.0038246, 0.0023886, 0.0040865, -0.0016905, 0.0014360
5: 0.0053121, 0.0070262, 0.0052965, 0.0072114, -0.0018993, 0.0017297
6: -0.0014960, -0.0007149, -0.0016110, -0.0007094, -0.0007867, 0.0008962
7: -0.0086714, -0.0074371, -0.0088047, -0.0074259, -0.0012455, 0.0013676
8: 0.0033434, 0.0072413, 0.0033087, 0.0076766, -0.0036234, 0.0032902
9: -0.0045368, -0.0022983, -0.0047785, -0.0022780, -0.0022588, 0.0024801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010118, upper bound: 0.0010576
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010118, upper bound: 0.0010833
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0006615, 0.0013609, 0.0006375, 0.0014257, -0.0007553, 0.0007144
1: 0.9930301, 0.9947939, 0.9928666, 0.9948542, -0.0017128, 0.0018160
2: -0.0068421, -0.0046465, -0.0071265, -0.0045897, -0.0022524, 0.0024799
3: 0.0035032, 0.0042347, 0.0034781, 0.0043025, -0.0007233, 0.0006806
4: 0.0023960, 0.0038246, 0.0023681, 0.0040494, -0.0016534, 0.0014565
5: 0.0053121, 0.0070262, 0.0052533, 0.0071851, -0.0018730, 0.0017730
6: -0.0014960, -0.0007149, -0.0015947, -0.0006941, -0.0008020, 0.0008799
7: -0.0086714, -0.0074371, -0.0087858, -0.0073948, -0.0012766, 0.0013487
8: 0.0033434, 0.0072413, 0.0032124, 0.0076149, -0.0035740, 0.0034158
9: -0.0045368, -0.0022983, -0.0047442, -0.0022215, -0.0023152, 0.0024459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010118, upper bound: 0.0010601
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010118, upper bound: 0.0010841
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0006615, 0.0013609, 0.0006489, 0.0014292, -0.0007587, 0.0007027
1: 0.9930301, 0.9947939, 0.9928579, 0.9948256, -0.0016831, 0.0018245
2: -0.0068421, -0.0046465, -0.0071416, -0.0046168, -0.0022253, 0.0024950
3: 0.0035032, 0.0042347, 0.0034900, 0.0043061, -0.0007265, 0.0006686
4: 0.0023960, 0.0038246, 0.0023814, 0.0040613, -0.0016653, 0.0014432
5: 0.0053121, 0.0070262, 0.0052813, 0.0071936, -0.0018815, 0.0017449
6: -0.0014960, -0.0007149, -0.0016000, -0.0007040, -0.0007920, 0.0008851
7: -0.0086714, -0.0074371, -0.0087919, -0.0074150, -0.0012564, 0.0013548
8: 0.0033434, 0.0072413, 0.0032749, 0.0076348, -0.0035871, 0.0033377
9: -0.0045368, -0.0022983, -0.0047553, -0.0022582, -0.0022786, 0.0024569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010104, upper bound: 0.0010585
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010104, upper bound: 0.0010851
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0006615, 0.0013609, 0.0006317, 0.0014181, -0.0007477, 0.0007200
1: 0.9930301, 0.9947939, 0.9928859, 0.9948689, -0.0017277, 0.0017977
2: -0.0068421, -0.0046465, -0.0070932, -0.0045761, -0.0022659, 0.0024466
3: 0.0035032, 0.0042347, 0.0034720, 0.0042945, -0.0007158, 0.0006875
4: 0.0023960, 0.0038246, 0.0023615, 0.0040230, -0.0016271, 0.0014632
5: 0.0053121, 0.0070262, 0.0052392, 0.0071665, -0.0018544, 0.0017870
6: -0.0014960, -0.0007149, -0.0015832, -0.0006891, -0.0008069, 0.0008683
7: -0.0086714, -0.0074371, -0.0087724, -0.0073846, -0.0012867, 0.0013353
8: 0.0033434, 0.0072413, 0.0031812, 0.0075711, -0.0035285, 0.0034587
9: -0.0045368, -0.0022983, -0.0047199, -0.0022032, -0.0023336, 0.0024216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010104, upper bound: 0.0010606
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010104, upper bound: 0.0010860
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006660, 0.0013728, 0.0006592, 0.0014364, -0.0007613, 0.0007046
1: 0.9930000, 0.9947824, 0.9928398, 0.9947996, -0.0016879, 0.0018305
2: -0.0068944, -0.0046572, -0.0071732, -0.0046411, -0.0022533, 0.0025160
3: 0.0035079, 0.0042471, 0.0035008, 0.0043136, -0.0007298, 0.0006701
4: 0.0024012, 0.0038660, 0.0023933, 0.0040863, -0.0016851, 0.0014727
5: 0.0053231, 0.0070555, 0.0053064, 0.0072112, -0.0018881, 0.0017490
6: -0.0015142, -0.0007188, -0.0016110, -0.0007129, -0.0008013, 0.0008922
7: -0.0086924, -0.0074451, -0.0088046, -0.0074330, -0.0012594, 0.0013595
8: 0.0033680, 0.0073100, 0.0033308, 0.0076763, -0.0036689, 0.0033125
9: -0.0045749, -0.0023128, -0.0047783, -0.0022910, -0.0022840, 0.0024656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010039, upper bound: 0.0010491
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010039, upper bound: 0.0010491
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006660, 0.0013728, 0.0006415, 0.0014257, -0.0007506, 0.0007224
1: 0.9930000, 0.9947824, 0.9928668, 0.9948441, -0.0017337, 0.0018037
2: -0.0068944, -0.0046572, -0.0071263, -0.0045994, -0.0022950, 0.0024691
3: 0.0035079, 0.0042471, 0.0034823, 0.0043024, -0.0007189, 0.0006891
4: 0.0024012, 0.0038660, 0.0023729, 0.0040492, -0.0016480, 0.0014931
5: 0.0053231, 0.0070555, 0.0052633, 0.0071850, -0.0018619, 0.0017922
6: -0.0015142, -0.0007188, -0.0015947, -0.0006976, -0.0008166, 0.0008759
7: -0.0086924, -0.0074451, -0.0087857, -0.0074020, -0.0012905, 0.0013406
8: 0.0033680, 0.0073100, 0.0032347, 0.0076147, -0.0036141, 0.0034388
9: -0.0045749, -0.0023128, -0.0047441, -0.0022346, -0.0023403, 0.0024313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010039, upper bound: 0.0010512
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010039, upper bound: 0.0010512
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006532, 0.0013553, 0.0006551, 0.0014364, -0.0007743, 0.0006912
1: 0.9930442, 0.9948147, 0.9928396, 0.9948098, -0.0016544, 0.0018643
2: -0.0068175, -0.0046269, -0.0071734, -0.0046315, -0.0021861, 0.0025465
3: 0.0034945, 0.0042288, 0.0034965, 0.0043137, -0.0007438, 0.0006567
4: 0.0023863, 0.0038052, 0.0023886, 0.0040865, -0.0017001, 0.0014166
5: 0.0052917, 0.0070125, 0.0052965, 0.0072114, -0.0019196, 0.0017160
6: -0.0014875, -0.0007077, -0.0016110, -0.0007094, -0.0007782, 0.0009034
7: -0.0086615, -0.0074224, -0.0088047, -0.0074259, -0.0012356, 0.0013822
8: 0.0032981, 0.0072090, 0.0033087, 0.0076766, -0.0037048, 0.0032819
9: -0.0045188, -0.0022718, -0.0047785, -0.0022780, -0.0022409, 0.0025067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010118, upper bound: 0.0010569
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010118, upper bound: 0.0010843
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006532, 0.0013553, 0.0006375, 0.0014257, -0.0007636, 0.0007089
1: 0.9930442, 0.9948147, 0.9928666, 0.9948542, -0.0017001, 0.0018379
2: -0.0068175, -0.0046269, -0.0071265, -0.0045897, -0.0022278, 0.0024996
3: 0.0034945, 0.0042288, 0.0034781, 0.0043025, -0.0007329, 0.0006756
4: 0.0023863, 0.0038052, 0.0023681, 0.0040494, -0.0016630, 0.0014371
5: 0.0052917, 0.0070125, 0.0052533, 0.0071851, -0.0018934, 0.0017593
6: -0.0014875, -0.0007077, -0.0015947, -0.0006941, -0.0007934, 0.0008871
7: -0.0086615, -0.0074224, -0.0087858, -0.0073948, -0.0012668, 0.0013633
8: 0.0032981, 0.0072090, 0.0032124, 0.0076149, -0.0036554, 0.0034075
9: -0.0045188, -0.0022718, -0.0047442, -0.0022215, -0.0022973, 0.0024725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010118, upper bound: 0.0010592
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010118, upper bound: 0.0010843
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006660, 0.0013728, 0.0006530, 0.0014291, -0.0007542, 0.0007107
1: 0.9930000, 0.9947824, 0.9928580, 0.9948150, -0.0017040, 0.0018131
2: -0.0068944, -0.0046572, -0.0071414, -0.0046266, -0.0022678, 0.0024842
3: 0.0035079, 0.0042471, 0.0034943, 0.0043060, -0.0007224, 0.0006771
4: 0.0024012, 0.0038660, 0.0023862, 0.0040612, -0.0016599, 0.0014798
5: 0.0053231, 0.0070555, 0.0052914, 0.0071935, -0.0018703, 0.0017640
6: -0.0015142, -0.0007188, -0.0015999, -0.0007076, -0.0008066, 0.0008812
7: -0.0086924, -0.0074451, -0.0087918, -0.0074222, -0.0012702, 0.0013467
8: 0.0033680, 0.0073100, 0.0032974, 0.0076345, -0.0036392, 0.0033723
9: -0.0045749, -0.0023128, -0.0047551, -0.0022714, -0.0023036, 0.0024423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010028, upper bound: 0.0010498
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010028, upper bound: 0.0010574
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006660, 0.0013728, 0.0006359, 0.0014181, -0.0007431, 0.0007279
1: 0.9930000, 0.9947824, 0.9928859, 0.9948581, -0.0017490, 0.0017858
2: -0.0068944, -0.0046572, -0.0070929, -0.0045860, -0.0023084, 0.0024357
3: 0.0035079, 0.0042471, 0.0034764, 0.0042945, -0.0007115, 0.0006960
4: 0.0024012, 0.0038660, 0.0023663, 0.0040229, -0.0016217, 0.0014997
5: 0.0053231, 0.0070555, 0.0052494, 0.0071664, -0.0018433, 0.0018060
6: -0.0015142, -0.0007188, -0.0015831, -0.0006927, -0.0008215, 0.0008643
7: -0.0086924, -0.0074451, -0.0087723, -0.0073920, -0.0013004, 0.0013272
8: 0.0033680, 0.0073100, 0.0032039, 0.0075709, -0.0035878, 0.0035021
9: -0.0045749, -0.0023128, -0.0047198, -0.0022165, -0.0023584, 0.0024070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010028, upper bound: 0.0010517
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010028, upper bound: 0.0010591
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006532, 0.0013553, 0.0006489, 0.0014292, -0.0007671, 0.0006973
1: 0.9930442, 0.9948147, 0.9928579, 0.9948256, -0.0016701, 0.0018464
2: -0.0068175, -0.0046269, -0.0071416, -0.0046168, -0.0022007, 0.0025147
3: 0.0034945, 0.0042288, 0.0034900, 0.0043061, -0.0007363, 0.0006634
4: 0.0023863, 0.0038052, 0.0023814, 0.0040613, -0.0016750, 0.0014238
5: 0.0052917, 0.0070125, 0.0052813, 0.0071936, -0.0019018, 0.0017312
6: -0.0014875, -0.0007077, -0.0016000, -0.0007040, -0.0007835, 0.0008923
7: -0.0086615, -0.0074224, -0.0087919, -0.0074150, -0.0012466, 0.0013694
8: 0.0032981, 0.0072090, 0.0032749, 0.0076348, -0.0036790, 0.0033330
9: -0.0045188, -0.0022718, -0.0047553, -0.0022582, -0.0022607, 0.0024835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010105, upper bound: 0.0010589
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010105, upper bound: 0.0010902
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006532, 0.0013553, 0.0006317, 0.0014181, -0.0007561, 0.0007146
1: 0.9930442, 0.9948147, 0.9928859, 0.9948689, -0.0017151, 0.0018194
2: -0.0068175, -0.0046269, -0.0070932, -0.0045761, -0.0022414, 0.0024663
3: 0.0034945, 0.0042288, 0.0034720, 0.0042945, -0.0007255, 0.0006821
4: 0.0023863, 0.0038052, 0.0023615, 0.0040230, -0.0016367, 0.0014438
5: 0.0052917, 0.0070125, 0.0052392, 0.0071665, -0.0018748, 0.0017733
6: -0.0014875, -0.0007077, -0.0015832, -0.0006891, -0.0007984, 0.0008755
7: -0.0086615, -0.0074224, -0.0087724, -0.0073846, -0.0012769, 0.0013499
8: 0.0032981, 0.0072090, 0.0031812, 0.0075711, -0.0036343, 0.0034629
9: -0.0045188, -0.0022718, -0.0047199, -0.0022032, -0.0023156, 0.0024482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010105, upper bound: 0.0010608
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010105, upper bound: 0.0010908
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006561, 0.0013691, 0.0006592, 0.0014364, -0.0007712, 0.0007007
1: 0.9930096, 0.9948073, 0.9928398, 0.9947996, -0.0016765, 0.0018557
2: -0.0068779, -0.0046340, -0.0071732, -0.0046411, -0.0022368, 0.0025392
3: 0.0034976, 0.0042432, 0.0035008, 0.0043136, -0.0007398, 0.0006654
4: 0.0023898, 0.0038529, 0.0023933, 0.0040863, -0.0016965, 0.0014596
5: 0.0052991, 0.0070462, 0.0053064, 0.0072112, -0.0019122, 0.0017398
6: -0.0015085, -0.0007103, -0.0016110, -0.0007129, -0.0007956, 0.0009007
7: -0.0086858, -0.0074277, -0.0088046, -0.0074330, -0.0012527, 0.0013768
8: 0.0033144, 0.0072883, 0.0033308, 0.0076763, -0.0037272, 0.0032804
9: -0.0045629, -0.0022813, -0.0047783, -0.0022910, -0.0022719, 0.0024970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010735
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010833
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006561, 0.0013691, 0.0006530, 0.0014291, -0.0007640, 0.0007067
1: 0.9930096, 0.9948073, 0.9928580, 0.9948150, -0.0016923, 0.0018378
2: -0.0068779, -0.0046340, -0.0071414, -0.0046266, -0.0022513, 0.0025074
3: 0.0034976, 0.0042432, 0.0034943, 0.0043060, -0.0007324, 0.0006722
4: 0.0023898, 0.0038529, 0.0023862, 0.0040612, -0.0016713, 0.0014667
5: 0.0052991, 0.0070462, 0.0052914, 0.0071935, -0.0018944, 0.0017548
6: -0.0015085, -0.0007103, -0.0015999, -0.0007076, -0.0008009, 0.0008897
7: -0.0086858, -0.0074277, -0.0087918, -0.0074222, -0.0012635, 0.0013640
8: 0.0033144, 0.0072883, 0.0032974, 0.0076345, -0.0036910, 0.0033273
9: -0.0045629, -0.0022813, -0.0047551, -0.0022714, -0.0022915, 0.0024738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010741
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010741
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006440, 0.0013500, 0.0006551, 0.0014364, -0.0007836, 0.0006858
1: 0.9930576, 0.9948378, 0.9928396, 0.9948098, -0.0016405, 0.0018873
2: -0.0067942, -0.0046051, -0.0071734, -0.0046315, -0.0021627, 0.0025683
3: 0.0034849, 0.0042232, 0.0034965, 0.0043137, -0.0007531, 0.0006508
4: 0.0023757, 0.0037868, 0.0023886, 0.0040865, -0.0017108, 0.0013982
5: 0.0052692, 0.0069995, 0.0052965, 0.0072114, -0.0019421, 0.0017030
6: -0.0014794, -0.0006997, -0.0016110, -0.0007094, -0.0007700, 0.0009113
7: -0.0086521, -0.0074062, -0.0088047, -0.0074259, -0.0012262, 0.0013984
8: 0.0032480, 0.0071784, 0.0033087, 0.0076766, -0.0037595, 0.0032458
9: -0.0045018, -0.0022424, -0.0047785, -0.0022780, -0.0022238, 0.0025361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010108, upper bound: 0.0010764
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010108, upper bound: 0.0010967
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006440, 0.0013500, 0.0006489, 0.0014292, -0.0007763, 0.0006919
1: 0.9930576, 0.9948378, 0.9928579, 0.9948256, -0.0016565, 0.0018694
2: -0.0067942, -0.0046051, -0.0071416, -0.0046168, -0.0021774, 0.0025364
3: 0.0034849, 0.0042232, 0.0034900, 0.0043061, -0.0007455, 0.0006577
4: 0.0023757, 0.0037868, 0.0023814, 0.0040613, -0.0016857, 0.0014054
5: 0.0052692, 0.0069995, 0.0052813, 0.0071936, -0.0019243, 0.0017181
6: -0.0014794, -0.0006997, -0.0016000, -0.0007040, -0.0007754, 0.0009003
7: -0.0086521, -0.0074062, -0.0087919, -0.0074150, -0.0012372, 0.0013856
8: 0.0032480, 0.0071784, 0.0032749, 0.0076348, -0.0037233, 0.0032933
9: -0.0045018, -0.0022424, -0.0047553, -0.0022582, -0.0022436, 0.0025129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010108, upper bound: 0.0010768
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010108, upper bound: 0.0010975
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006561, 0.0013691, 0.0006415, 0.0014257, -0.0007602, 0.0007181
1: 0.9930096, 0.9948073, 0.9928668, 0.9948441, -0.0017186, 0.0018252
2: -0.0068779, -0.0046340, -0.0071263, -0.0045994, -0.0022785, 0.0024923
3: 0.0034976, 0.0042432, 0.0034823, 0.0043024, -0.0007262, 0.0006820
4: 0.0023898, 0.0038529, 0.0023729, 0.0040492, -0.0016594, 0.0014800
5: 0.0052991, 0.0070462, 0.0052633, 0.0071850, -0.0018859, 0.0017830
6: -0.0015085, -0.0007103, -0.0015947, -0.0006976, -0.0008108, 0.0008844
7: -0.0086858, -0.0074277, -0.0087857, -0.0074020, -0.0012838, 0.0013580
8: 0.0033144, 0.0072883, 0.0032347, 0.0076147, -0.0035961, 0.0033361
9: -0.0045629, -0.0022813, -0.0047441, -0.0022346, -0.0023283, 0.0024628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010308, upper bound: 0.0010957
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010308, upper bound: 0.0011040
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006561, 0.0013691, 0.0006359, 0.0014181, -0.0007527, 0.0007237
1: 0.9930096, 0.9948073, 0.9928859, 0.9948581, -0.0017332, 0.0018073
2: -0.0068779, -0.0046340, -0.0070929, -0.0045860, -0.0022919, 0.0024590
3: 0.0034976, 0.0042432, 0.0034764, 0.0042945, -0.0007190, 0.0006886
4: 0.0023898, 0.0038529, 0.0023663, 0.0040229, -0.0016331, 0.0014866
5: 0.0052991, 0.0070462, 0.0052494, 0.0071664, -0.0018673, 0.0017968
6: -0.0015085, -0.0007103, -0.0015831, -0.0006927, -0.0008157, 0.0008729
7: -0.0086858, -0.0074277, -0.0087723, -0.0073920, -0.0012938, 0.0013446
8: 0.0033144, 0.0072883, 0.0032039, 0.0075709, -0.0035609, 0.0033824
9: -0.0045629, -0.0022813, -0.0047198, -0.0022165, -0.0023463, 0.0024385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010308, upper bound: 0.0010957
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010308, upper bound: 0.0011040
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006440, 0.0013500, 0.0006375, 0.0014257, -0.0007726, 0.0007033
1: 0.9930576, 0.9948378, 0.9928666, 0.9948542, -0.0016827, 0.0018572
2: -0.0067942, -0.0046051, -0.0071265, -0.0045897, -0.0022045, 0.0025213
3: 0.0034849, 0.0042232, 0.0034781, 0.0043025, -0.0007396, 0.0006673
4: 0.0023757, 0.0037868, 0.0023681, 0.0040494, -0.0016737, 0.0014187
5: 0.0052692, 0.0069995, 0.0052533, 0.0071851, -0.0019159, 0.0017462
6: -0.0014794, -0.0006997, -0.0015947, -0.0006941, -0.0007853, 0.0008950
7: -0.0086521, -0.0074062, -0.0087858, -0.0073948, -0.0012574, 0.0013796
8: 0.0032480, 0.0071784, 0.0032124, 0.0076149, -0.0036291, 0.0032985
9: -0.0045018, -0.0022424, -0.0047442, -0.0022215, -0.0022803, 0.0025019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0010968
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0011125
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006440, 0.0013500, 0.0006317, 0.0014181, -0.0007650, 0.0007090
1: 0.9930576, 0.9948378, 0.9928859, 0.9948689, -0.0016976, 0.0018391
2: -0.0067942, -0.0046051, -0.0070932, -0.0045761, -0.0022180, 0.0024880
3: 0.0034849, 0.0042232, 0.0034720, 0.0042945, -0.0007322, 0.0006740
4: 0.0023757, 0.0037868, 0.0023615, 0.0040230, -0.0016474, 0.0014253
5: 0.0052692, 0.0069995, 0.0052392, 0.0071665, -0.0018973, 0.0017603
6: -0.0014794, -0.0006997, -0.0015832, -0.0006891, -0.0007903, 0.0008835
7: -0.0086521, -0.0074062, -0.0087724, -0.0073846, -0.0012675, 0.0013662
8: 0.0032480, 0.0071784, 0.0031812, 0.0075711, -0.0035917, 0.0033454
9: -0.0045018, -0.0022424, -0.0047199, -0.0022032, -0.0022986, 0.0024776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0010968
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0011125
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006502, 0.0013620, 0.0006592, 0.0014364, -0.0007772, 0.0006937
1: 0.9930273, 0.9948221, 0.9928398, 0.9947996, -0.0016597, 0.0018722
2: -0.0068469, -0.0046198, -0.0071732, -0.0046411, -0.0022058, 0.0025533
3: 0.0034914, 0.0042358, 0.0035008, 0.0043136, -0.0007474, 0.0006584
4: 0.0023829, 0.0038284, 0.0023933, 0.0040863, -0.0017034, 0.0014351
5: 0.0052845, 0.0070289, 0.0053064, 0.0072112, -0.0019268, 0.0017225
6: -0.0014977, -0.0007051, -0.0016110, -0.0007129, -0.0007848, 0.0009058
7: -0.0086733, -0.0074172, -0.0088046, -0.0074330, -0.0012403, 0.0013874
8: 0.0032819, 0.0072476, 0.0033308, 0.0076763, -0.0038012, 0.0032667
9: -0.0045402, -0.0022623, -0.0047783, -0.0022910, -0.0022493, 0.0025161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010041, upper bound: 0.0010692
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010041, upper bound: 0.0010787
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006368, 0.0013433, 0.0006551, 0.0014364, -0.0007907, 0.0006792
1: 0.9930744, 0.9948560, 0.9928396, 0.9948098, -0.0016250, 0.0019063
2: -0.0067649, -0.0045882, -0.0071734, -0.0046315, -0.0021334, 0.0025852
3: 0.0034774, 0.0042163, 0.0034965, 0.0043137, -0.0007617, 0.0006447
4: 0.0023674, 0.0037636, 0.0023886, 0.0040865, -0.0017191, 0.0013750
5: 0.0052517, 0.0069831, 0.0052965, 0.0072114, -0.0019597, 0.0016866
6: -0.0014693, -0.0006935, -0.0016110, -0.0007094, -0.0007599, 0.0009175
7: -0.0086403, -0.0073936, -0.0088047, -0.0074259, -0.0012145, 0.0014111
8: 0.0032089, 0.0071399, 0.0033087, 0.0076766, -0.0038357, 0.0032382
9: -0.0044804, -0.0022194, -0.0047785, -0.0022780, -0.0022025, 0.0025591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010120, upper bound: 0.0010733
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010120, upper bound: 0.0010975
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006502, 0.0013620, 0.0006415, 0.0014257, -0.0007662, 0.0007112
1: 0.9930273, 0.9948221, 0.9928668, 0.9948441, -0.0017022, 0.0018415
2: -0.0068469, -0.0046198, -0.0071263, -0.0045994, -0.0022475, 0.0025064
3: 0.0034914, 0.0042358, 0.0034823, 0.0043024, -0.0007340, 0.0006752
4: 0.0023829, 0.0038284, 0.0023729, 0.0040492, -0.0016663, 0.0014556
5: 0.0052845, 0.0070289, 0.0052633, 0.0071850, -0.0019005, 0.0017657
6: -0.0014977, -0.0007051, -0.0015947, -0.0006976, -0.0008001, 0.0008896
7: -0.0086733, -0.0074172, -0.0087857, -0.0074020, -0.0012714, 0.0013685
8: 0.0032819, 0.0072476, 0.0032347, 0.0076147, -0.0036733, 0.0033223
9: -0.0045402, -0.0022623, -0.0047441, -0.0022346, -0.0023057, 0.0024818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010331, upper bound: 0.0010903
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010331, upper bound: 0.0010977
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006368, 0.0013433, 0.0006375, 0.0014257, -0.0007797, 0.0006968
1: 0.9930744, 0.9948560, 0.9928666, 0.9948542, -0.0016676, 0.0018763
2: -0.0067649, -0.0045882, -0.0071265, -0.0045897, -0.0021752, 0.0025383
3: 0.0034774, 0.0042163, 0.0034781, 0.0043025, -0.0007483, 0.0006613
4: 0.0023674, 0.0037636, 0.0023681, 0.0040494, -0.0016820, 0.0013955
5: 0.0052517, 0.0069831, 0.0052533, 0.0071851, -0.0019335, 0.0017299
6: -0.0014693, -0.0006935, -0.0015947, -0.0006941, -0.0007752, 0.0009012
7: -0.0086403, -0.0073936, -0.0087858, -0.0073948, -0.0012456, 0.0013922
8: 0.0032089, 0.0071399, 0.0032124, 0.0076149, -0.0037085, 0.0032903
9: -0.0044804, -0.0022194, -0.0047442, -0.0022215, -0.0022589, 0.0025248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010398, upper bound: 0.0010926
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010398, upper bound: 0.0011105
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006502, 0.0013620, 0.0006530, 0.0014291, -0.0007701, 0.0006999
1: 0.9930273, 0.9948221, 0.9928580, 0.9948150, -0.0016767, 0.0018561
2: -0.0068469, -0.0046198, -0.0071414, -0.0046266, -0.0022203, 0.0025215
3: 0.0034914, 0.0042358, 0.0034943, 0.0043060, -0.0007408, 0.0006661
4: 0.0023829, 0.0038284, 0.0023862, 0.0040612, -0.0016783, 0.0014422
5: 0.0052845, 0.0070289, 0.0052914, 0.0071935, -0.0019090, 0.0017375
6: -0.0014977, -0.0007051, -0.0015999, -0.0007076, -0.0007901, 0.0008948
7: -0.0086733, -0.0074172, -0.0087918, -0.0074222, -0.0012511, 0.0013746
8: 0.0032819, 0.0072476, 0.0032974, 0.0076345, -0.0037827, 0.0033272
9: -0.0045402, -0.0022623, -0.0047551, -0.0022714, -0.0022689, 0.0024928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010704
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010029, upper bound: 0.0010792
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006368, 0.0013433, 0.0006489, 0.0014292, -0.0007835, 0.0006854
1: 0.9930744, 0.9948560, 0.9928579, 0.9948256, -0.0016418, 0.0018893
2: -0.0067649, -0.0045882, -0.0071416, -0.0046168, -0.0021481, 0.0025534
3: 0.0034774, 0.0042163, 0.0034900, 0.0043061, -0.0007545, 0.0006523
4: 0.0023674, 0.0037636, 0.0023814, 0.0040613, -0.0016940, 0.0013822
5: 0.0052517, 0.0069831, 0.0052813, 0.0071936, -0.0019419, 0.0017018
6: -0.0014693, -0.0006935, -0.0016000, -0.0007040, -0.0007653, 0.0009065
7: -0.0086403, -0.0073936, -0.0087919, -0.0074150, -0.0012254, 0.0013983
8: 0.0032089, 0.0071399, 0.0032749, 0.0076348, -0.0038208, 0.0032895
9: -0.0044804, -0.0022194, -0.0047553, -0.0022582, -0.0022223, 0.0025358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010108, upper bound: 0.0010748
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010108, upper bound: 0.0011021
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006502, 0.0013620, 0.0006359, 0.0014181, -0.0007588, 0.0007169
1: 0.9930273, 0.9948221, 0.9928859, 0.9948581, -0.0017181, 0.0018250
2: -0.0068469, -0.0046198, -0.0070929, -0.0045860, -0.0022609, 0.0024731
3: 0.0034914, 0.0042358, 0.0034764, 0.0042945, -0.0007271, 0.0006825
4: 0.0023829, 0.0038284, 0.0023663, 0.0040229, -0.0016400, 0.0014621
5: 0.0052845, 0.0070289, 0.0052494, 0.0071664, -0.0018819, 0.0017795
6: -0.0014977, -0.0007051, -0.0015831, -0.0006927, -0.0008050, 0.0008780
7: -0.0086733, -0.0074172, -0.0087723, -0.0073920, -0.0012813, 0.0013551
8: 0.0032819, 0.0072476, 0.0032039, 0.0075709, -0.0036469, 0.0033811
9: -0.0045402, -0.0022623, -0.0047198, -0.0022165, -0.0023237, 0.0024575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010308, upper bound: 0.0010903
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010308, upper bound: 0.0010978
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006368, 0.0013433, 0.0006317, 0.0014181, -0.0007723, 0.0007025
1: 0.9930744, 0.9948560, 0.9928859, 0.9948689, -0.0016834, 0.0018586
2: -0.0067649, -0.0045882, -0.0070932, -0.0045761, -0.0021888, 0.0025050
3: 0.0034774, 0.0042163, 0.0034720, 0.0042945, -0.0007412, 0.0006685
4: 0.0023674, 0.0037636, 0.0023615, 0.0040230, -0.0016557, 0.0014022
5: 0.0052517, 0.0069831, 0.0052392, 0.0071665, -0.0019149, 0.0017439
6: -0.0014693, -0.0006935, -0.0015832, -0.0006891, -0.0007801, 0.0008897
7: -0.0086403, -0.0073936, -0.0087724, -0.0073846, -0.0012557, 0.0013788
8: 0.0032089, 0.0071399, 0.0031812, 0.0075711, -0.0036863, 0.0033414
9: -0.0044804, -0.0022194, -0.0047199, -0.0022032, -0.0022773, 0.0025005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0010926
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010370, upper bound: 0.0011133
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0014364, 0.0006730, 0.0013796, -0.0007111, 0.0007543
1: 0.9928398, 0.9947996, 0.9929829, 0.9947646, -0.0018115, 0.0017022
2: -0.0071732, -0.0046411, -0.0069241, -0.0046738, -0.0024994, 0.0022830
3: 0.0035008, 0.0043136, 0.0035152, 0.0042542, -0.0006759, 0.0007209
4: 0.0023933, 0.0040863, 0.0024094, 0.0038894, -0.0014961, 0.0016769
5: 0.0053064, 0.0072112, 0.0053403, 0.0070721, -0.0017656, 0.0018709
6: -0.0016110, -0.0007129, -0.0015245, -0.0007249, -0.0008861, 0.0008116
7: -0.0088046, -0.0074330, -0.0087044, -0.0074575, -0.0013471, 0.0012713
8: 0.0033308, 0.0076763, 0.0034063, 0.0073490, -0.0033258, 0.0035903
9: -0.0047783, -0.0022910, -0.0045966, -0.0023352, -0.0024431, 0.0023056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010517, upper bound: 0.0010039
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010517, upper bound: 0.0010039
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0006415, 0.0014257, 0.0006730, 0.0013796, -0.0007289, 0.0007436
1: 0.9928668, 0.9948441, 0.9929829, 0.9947646, -0.0017846, 0.0017480
2: -0.0071263, -0.0045994, -0.0069241, -0.0046738, -0.0024524, 0.0023247
3: 0.0034823, 0.0043024, 0.0035152, 0.0042542, -0.0006949, 0.0007101
4: 0.0023729, 0.0040492, 0.0024094, 0.0038894, -0.0015166, 0.0016399
5: 0.0052633, 0.0071850, 0.0053403, 0.0070721, -0.0018088, 0.0018447
6: -0.0015947, -0.0006976, -0.0015245, -0.0007249, -0.0008698, 0.0008269
7: -0.0087857, -0.0074020, -0.0087044, -0.0074575, -0.0013283, 0.0013024
8: 0.0032347, 0.0076147, 0.0034063, 0.0073490, -0.0034521, 0.0035355
9: -0.0047441, -0.0022346, -0.0045966, -0.0023352, -0.0024089, 0.0023620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010517, upper bound: 0.0010202
time: 0.93 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010517, upper bound: 0.0010233
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0006530, 0.0014291, 0.0006730, 0.0013796, -0.0007172, 0.0007470
1: 0.9928580, 0.9948150, 0.9929829, 0.9947646, -0.0017936, 0.0017179
2: -0.0071414, -0.0046266, -0.0069241, -0.0046738, -0.0024675, 0.0022975
3: 0.0034943, 0.0043060, 0.0035152, 0.0042542, -0.0006828, 0.0007135
4: 0.0023862, 0.0040612, 0.0024094, 0.0038894, -0.0015032, 0.0016518
5: 0.0052914, 0.0071935, 0.0053403, 0.0070721, -0.0017806, 0.0018531
6: -0.0015999, -0.0007076, -0.0015245, -0.0007249, -0.0008751, 0.0008169
7: -0.0087918, -0.0074222, -0.0087044, -0.0074575, -0.0013343, 0.0012821
8: 0.0032974, 0.0076345, 0.0034063, 0.0073490, -0.0033728, 0.0035541
9: -0.0047551, -0.0022714, -0.0045966, -0.0023352, -0.0024199, 0.0023252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010523, upper bound: 0.0010029
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010523, upper bound: 0.0010118
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0006359, 0.0014181, 0.0006730, 0.0013796, -0.0007345, 0.0007360
1: 0.9928859, 0.9948581, 0.9929829, 0.9947646, -0.0017665, 0.0017626
2: -0.0070929, -0.0045860, -0.0069241, -0.0046738, -0.0024191, 0.0023381
3: 0.0034764, 0.0042945, 0.0035152, 0.0042542, -0.0007017, 0.0007027
4: 0.0023663, 0.0040229, 0.0024094, 0.0038894, -0.0015231, 0.0016135
5: 0.0052494, 0.0071664, 0.0053403, 0.0070721, -0.0018226, 0.0018261
6: -0.0015831, -0.0006927, -0.0015245, -0.0007249, -0.0008583, 0.0008318
7: -0.0087723, -0.0073920, -0.0087044, -0.0074575, -0.0013149, 0.0013124
8: 0.0032039, 0.0075709, 0.0034063, 0.0073490, -0.0034932, 0.0034973
9: -0.0047198, -0.0022165, -0.0045966, -0.0023352, -0.0023846, 0.0023801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010523, upper bound: 0.0010176
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010523, upper bound: 0.0010225
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0006551, 0.0014364, 0.0006615, 0.0013609, -0.0006966, 0.0007660
1: 0.9928396, 0.9948098, 0.9930301, 0.9947939, -0.0018423, 0.0016671
2: -0.0071734, -0.0046315, -0.0068421, -0.0046465, -0.0025269, 0.0022106
3: 0.0034965, 0.0043137, 0.0035032, 0.0042347, -0.0006617, 0.0007341
4: 0.0023886, 0.0040865, 0.0023960, 0.0038246, -0.0014360, 0.0016905
5: 0.0052965, 0.0072114, 0.0053121, 0.0070262, -0.0017297, 0.0018993
6: -0.0016110, -0.0007094, -0.0014960, -0.0007149, -0.0008962, 0.0007867
7: -0.0088047, -0.0074259, -0.0086714, -0.0074371, -0.0013676, 0.0012455
8: 0.0033087, 0.0076766, 0.0033434, 0.0072413, -0.0032902, 0.0036234
9: -0.0047785, -0.0022780, -0.0045368, -0.0022983, -0.0024801, 0.0022588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010517, upper bound: 0.0010118
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010517, upper bound: 0.0010297
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006615, 0.0013609, -0.0007144, 0.0007553
1: 0.9928666, 0.9948542, 0.9930301, 0.9947939, -0.0018159, 0.0017128
2: -0.0071265, -0.0045897, -0.0068421, -0.0046465, -0.0024799, 0.0022524
3: 0.0034781, 0.0043025, 0.0035032, 0.0042347, -0.0006806, 0.0007233
4: 0.0023681, 0.0040494, 0.0023960, 0.0038246, -0.0014565, 0.0016534
5: 0.0052533, 0.0071851, 0.0053121, 0.0070262, -0.0017730, 0.0018730
6: -0.0015947, -0.0006941, -0.0014960, -0.0007149, -0.0008799, 0.0008020
7: -0.0087858, -0.0073948, -0.0086714, -0.0074371, -0.0013487, 0.0012766
8: 0.0032124, 0.0076149, 0.0033434, 0.0072413, -0.0034158, 0.0035740
9: -0.0047442, -0.0022215, -0.0045368, -0.0022983, -0.0024459, 0.0023152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010517, upper bound: 0.0010254
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010517, upper bound: 0.0010404
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0006489, 0.0014292, 0.0006615, 0.0013609, -0.0007027, 0.0007587
1: 0.9928579, 0.9948256, 0.9930301, 0.9947939, -0.0018245, 0.0016831
2: -0.0071416, -0.0046168, -0.0068421, -0.0046465, -0.0024950, 0.0022253
3: 0.0034900, 0.0043061, 0.0035032, 0.0042347, -0.0006686, 0.0007265
4: 0.0023814, 0.0040613, 0.0023960, 0.0038246, -0.0014432, 0.0016653
5: 0.0052813, 0.0071936, 0.0053121, 0.0070262, -0.0017449, 0.0018815
6: -0.0016000, -0.0007040, -0.0014960, -0.0007149, -0.0008851, 0.0007920
7: -0.0087919, -0.0074150, -0.0086714, -0.0074371, -0.0013548, 0.0012564
8: 0.0032749, 0.0076348, 0.0033434, 0.0072413, -0.0033377, 0.0035871
9: -0.0047553, -0.0022582, -0.0045368, -0.0022983, -0.0024569, 0.0022786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010523, upper bound: 0.0010104
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010523, upper bound: 0.0010325
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006615, 0.0013609, -0.0007200, 0.0007477
1: 0.9928859, 0.9948689, 0.9930301, 0.9947939, -0.0017977, 0.0017277
2: -0.0070932, -0.0045761, -0.0068421, -0.0046465, -0.0024466, 0.0022659
3: 0.0034720, 0.0042945, 0.0035032, 0.0042347, -0.0006875, 0.0007158
4: 0.0023615, 0.0040230, 0.0023960, 0.0038246, -0.0014632, 0.0016271
5: 0.0052392, 0.0071665, 0.0053121, 0.0070262, -0.0017870, 0.0018544
6: -0.0015832, -0.0006891, -0.0014960, -0.0007149, -0.0008683, 0.0008069
7: -0.0087724, -0.0073846, -0.0086714, -0.0074371, -0.0013353, 0.0012867
8: 0.0031812, 0.0075711, 0.0033434, 0.0072413, -0.0034587, 0.0035285
9: -0.0047199, -0.0022032, -0.0045368, -0.0022983, -0.0024216, 0.0023336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010523, upper bound: 0.0010234
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010523, upper bound: 0.0010412
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0014364, 0.0006660, 0.0013728, -0.0007046, 0.0007613
1: 0.9928398, 0.9947996, 0.9930000, 0.9947824, -0.0018305, 0.0016879
2: -0.0071732, -0.0046411, -0.0068944, -0.0046572, -0.0025160, 0.0022533
3: 0.0035008, 0.0043136, 0.0035079, 0.0042471, -0.0006701, 0.0007298
4: 0.0023933, 0.0040863, 0.0024012, 0.0038660, -0.0014727, 0.0016851
5: 0.0053064, 0.0072112, 0.0053231, 0.0070555, -0.0017490, 0.0018881
6: -0.0016110, -0.0007129, -0.0015142, -0.0007188, -0.0008922, 0.0008013
7: -0.0088046, -0.0074330, -0.0086924, -0.0074451, -0.0013595, 0.0012594
8: 0.0033308, 0.0076763, 0.0033680, 0.0073100, -0.0033125, 0.0036689
9: -0.0047783, -0.0022910, -0.0045749, -0.0023128, -0.0024656, 0.0022840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010491, upper bound: 0.0010039
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010491, upper bound: 0.0010099
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006415, 0.0014257, 0.0006660, 0.0013728, -0.0007224, 0.0007506
1: 0.9928668, 0.9948441, 0.9930000, 0.9947824, -0.0018037, 0.0017337
2: -0.0071263, -0.0045994, -0.0068944, -0.0046572, -0.0024691, 0.0022950
3: 0.0034823, 0.0043024, 0.0035079, 0.0042471, -0.0006891, 0.0007189
4: 0.0023729, 0.0040492, 0.0024012, 0.0038660, -0.0014931, 0.0016480
5: 0.0052633, 0.0071850, 0.0053231, 0.0070555, -0.0017922, 0.0018619
6: -0.0015947, -0.0006976, -0.0015142, -0.0007188, -0.0008759, 0.0008166
7: -0.0087857, -0.0074020, -0.0086924, -0.0074451, -0.0013406, 0.0012905
8: 0.0032347, 0.0076147, 0.0033680, 0.0073100, -0.0034388, 0.0036141
9: -0.0047441, -0.0022346, -0.0045749, -0.0023128, -0.0024313, 0.0023403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010491, upper bound: 0.0010202
time: 1.00 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010491, upper bound: 0.0010099
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006551, 0.0014364, 0.0006532, 0.0013553, -0.0006912, 0.0007743
1: 0.9928396, 0.9948098, 0.9930442, 0.9948147, -0.0018643, 0.0016544
2: -0.0071734, -0.0046315, -0.0068175, -0.0046269, -0.0025465, 0.0021861
3: 0.0034965, 0.0043137, 0.0034945, 0.0042288, -0.0006567, 0.0007438
4: 0.0023886, 0.0040865, 0.0023863, 0.0038052, -0.0014166, 0.0017001
5: 0.0052965, 0.0072114, 0.0052917, 0.0070125, -0.0017160, 0.0019196
6: -0.0016110, -0.0007094, -0.0014875, -0.0007077, -0.0009034, 0.0007782
7: -0.0088047, -0.0074259, -0.0086615, -0.0074224, -0.0013822, 0.0012356
8: 0.0033087, 0.0076766, 0.0032981, 0.0072090, -0.0032819, 0.0037048
9: -0.0047785, -0.0022780, -0.0045188, -0.0022718, -0.0025067, 0.0022409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010491, upper bound: 0.0010118
time: 1.00 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010491, upper bound: 0.0010297
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006532, 0.0013553, -0.0007089, 0.0007636
1: 0.9928666, 0.9948542, 0.9930442, 0.9948147, -0.0018379, 0.0017001
2: -0.0071265, -0.0045897, -0.0068175, -0.0046269, -0.0024996, 0.0022278
3: 0.0034781, 0.0043025, 0.0034945, 0.0042288, -0.0006756, 0.0007329
4: 0.0023681, 0.0040494, 0.0023863, 0.0038052, -0.0014371, 0.0016630
5: 0.0052533, 0.0071851, 0.0052917, 0.0070125, -0.0017593, 0.0018934
6: -0.0015947, -0.0006941, -0.0014875, -0.0007077, -0.0008871, 0.0007934
7: -0.0087858, -0.0073948, -0.0086615, -0.0074224, -0.0013633, 0.0012668
8: 0.0032124, 0.0076149, 0.0032981, 0.0072090, -0.0034075, 0.0036554
9: -0.0047442, -0.0022215, -0.0045188, -0.0022718, -0.0024725, 0.0022973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010491, upper bound: 0.0010254
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010491, upper bound: 0.0010297
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006530, 0.0014291, 0.0006660, 0.0013728, -0.0007107, 0.0007542
1: 0.9928580, 0.9948150, 0.9930000, 0.9947824, -0.0018131, 0.0017040
2: -0.0071414, -0.0046266, -0.0068944, -0.0046572, -0.0024842, 0.0022678
3: 0.0034943, 0.0043060, 0.0035079, 0.0042471, -0.0006771, 0.0007224
4: 0.0023862, 0.0040612, 0.0024012, 0.0038660, -0.0014798, 0.0016599
5: 0.0052914, 0.0071935, 0.0053231, 0.0070555, -0.0017640, 0.0018703
6: -0.0015999, -0.0007076, -0.0015142, -0.0007188, -0.0008812, 0.0008066
7: -0.0087918, -0.0074222, -0.0086924, -0.0074451, -0.0013467, 0.0012702
8: 0.0032974, 0.0076345, 0.0033680, 0.0073100, -0.0033723, 0.0036392
9: -0.0047551, -0.0022714, -0.0045749, -0.0023128, -0.0024423, 0.0023036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010510, upper bound: 0.0010029
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010510, upper bound: 0.0010029
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006359, 0.0014181, 0.0006660, 0.0013728, -0.0007279, 0.0007431
1: 0.9928859, 0.9948581, 0.9930000, 0.9947824, -0.0017858, 0.0017490
2: -0.0070929, -0.0045860, -0.0068944, -0.0046572, -0.0024357, 0.0023084
3: 0.0034764, 0.0042945, 0.0035079, 0.0042471, -0.0006960, 0.0007115
4: 0.0023663, 0.0040229, 0.0024012, 0.0038660, -0.0014997, 0.0016217
5: 0.0052494, 0.0071664, 0.0053231, 0.0070555, -0.0018060, 0.0018433
6: -0.0015831, -0.0006927, -0.0015142, -0.0007188, -0.0008643, 0.0008215
7: -0.0087723, -0.0073920, -0.0086924, -0.0074451, -0.0013272, 0.0013004
8: 0.0032039, 0.0075709, 0.0033680, 0.0073100, -0.0035021, 0.0035878
9: -0.0047198, -0.0022165, -0.0045749, -0.0023128, -0.0024070, 0.0023584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010510, upper bound: 0.0010176
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010510, upper bound: 0.0010225
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006489, 0.0014292, 0.0006532, 0.0013553, -0.0006973, 0.0007671
1: 0.9928579, 0.9948256, 0.9930442, 0.9948147, -0.0018464, 0.0016701
2: -0.0071416, -0.0046168, -0.0068175, -0.0046269, -0.0025147, 0.0022007
3: 0.0034900, 0.0043061, 0.0034945, 0.0042288, -0.0006634, 0.0007363
4: 0.0023814, 0.0040613, 0.0023863, 0.0038052, -0.0014238, 0.0016750
5: 0.0052813, 0.0071936, 0.0052917, 0.0070125, -0.0017312, 0.0019018
6: -0.0016000, -0.0007040, -0.0014875, -0.0007077, -0.0008923, 0.0007835
7: -0.0087919, -0.0074150, -0.0086615, -0.0074224, -0.0013694, 0.0012466
8: 0.0032749, 0.0076348, 0.0032981, 0.0072090, -0.0033330, 0.0036790
9: -0.0047553, -0.0022582, -0.0045188, -0.0022718, -0.0024835, 0.0022607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010510, upper bound: 0.0010104
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010510, upper bound: 0.0010343
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006532, 0.0013553, -0.0007146, 0.0007561
1: 0.9928859, 0.9948689, 0.9930442, 0.9948147, -0.0018194, 0.0017151
2: -0.0070932, -0.0045761, -0.0068175, -0.0046269, -0.0024663, 0.0022414
3: 0.0034720, 0.0042945, 0.0034945, 0.0042288, -0.0006821, 0.0007255
4: 0.0023615, 0.0040230, 0.0023863, 0.0038052, -0.0014438, 0.0016367
5: 0.0052392, 0.0071665, 0.0052917, 0.0070125, -0.0017733, 0.0018748
6: -0.0015832, -0.0006891, -0.0014875, -0.0007077, -0.0008755, 0.0007984
7: -0.0087724, -0.0073846, -0.0086615, -0.0074224, -0.0013499, 0.0012769
8: 0.0031812, 0.0075711, 0.0032981, 0.0072090, -0.0034629, 0.0036343
9: -0.0047199, -0.0022032, -0.0045188, -0.0022718, -0.0024482, 0.0023156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010510, upper bound: 0.0010104
time: 0.93 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010510, upper bound: 0.0010428
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0014364, 0.0006561, 0.0013691, -0.0007007, 0.0007712
1: 0.9928398, 0.9947996, 0.9930096, 0.9948073, -0.0018557, 0.0016765
2: -0.0071732, -0.0046411, -0.0068779, -0.0046340, -0.0025392, 0.0022368
3: 0.0035008, 0.0043136, 0.0034976, 0.0042432, -0.0006654, 0.0007398
4: 0.0023933, 0.0040863, 0.0023898, 0.0038529, -0.0014596, 0.0016965
5: 0.0053064, 0.0072112, 0.0052991, 0.0070462, -0.0017398, 0.0019122
6: -0.0016110, -0.0007129, -0.0015085, -0.0007103, -0.0009007, 0.0007956
7: -0.0088046, -0.0074330, -0.0086858, -0.0074277, -0.0013768, 0.0012527
8: 0.0033308, 0.0076763, 0.0033144, 0.0072883, -0.0032804, 0.0037272
9: -0.0047783, -0.0022910, -0.0045629, -0.0022813, -0.0024970, 0.0022719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010029
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010029
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006530, 0.0014291, 0.0006561, 0.0013691, -0.0007067, 0.0007640
1: 0.9928580, 0.9948150, 0.9930096, 0.9948073, -0.0018378, 0.0016923
2: -0.0071414, -0.0046266, -0.0068779, -0.0046340, -0.0025074, 0.0022513
3: 0.0034943, 0.0043060, 0.0034976, 0.0042432, -0.0006722, 0.0007324
4: 0.0023862, 0.0040612, 0.0023898, 0.0038529, -0.0014667, 0.0016713
5: 0.0052914, 0.0071935, 0.0052991, 0.0070462, -0.0017548, 0.0018944
6: -0.0015999, -0.0007076, -0.0015085, -0.0007103, -0.0008897, 0.0008009
7: -0.0087918, -0.0074222, -0.0086858, -0.0074277, -0.0013640, 0.0012635
8: 0.0032974, 0.0076345, 0.0033144, 0.0072883, -0.0033273, 0.0036910
9: -0.0047551, -0.0022714, -0.0045629, -0.0022813, -0.0024738, 0.0022915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010032
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010121
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006551, 0.0014364, 0.0006440, 0.0013500, -0.0006858, 0.0007836
1: 0.9928396, 0.9948098, 0.9930576, 0.9948378, -0.0018872, 0.0016405
2: -0.0071734, -0.0046315, -0.0067942, -0.0046051, -0.0025683, 0.0021627
3: 0.0034965, 0.0043137, 0.0034849, 0.0042232, -0.0006508, 0.0007531
4: 0.0023886, 0.0040865, 0.0023757, 0.0037868, -0.0013982, 0.0017108
5: 0.0052965, 0.0072114, 0.0052692, 0.0069995, -0.0017030, 0.0019421
6: -0.0016110, -0.0007094, -0.0014794, -0.0006997, -0.0009113, 0.0007700
7: -0.0088047, -0.0074259, -0.0086521, -0.0074062, -0.0013984, 0.0012262
8: 0.0033087, 0.0076766, 0.0032480, 0.0071784, -0.0032458, 0.0037595
9: -0.0047785, -0.0022780, -0.0045018, -0.0022424, -0.0025361, 0.0022238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010108
time: 1.18 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010029
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006489, 0.0014292, 0.0006440, 0.0013500, -0.0006919, 0.0007763
1: 0.9928579, 0.9948256, 0.9930576, 0.9948378, -0.0018694, 0.0016565
2: -0.0071416, -0.0046168, -0.0067942, -0.0046051, -0.0025364, 0.0021774
3: 0.0034900, 0.0043061, 0.0034849, 0.0042232, -0.0006577, 0.0007455
4: 0.0023814, 0.0040613, 0.0023757, 0.0037868, -0.0014054, 0.0016857
5: 0.0052813, 0.0071936, 0.0052692, 0.0069995, -0.0017181, 0.0019243
6: -0.0016000, -0.0007040, -0.0014794, -0.0006997, -0.0009003, 0.0007754
7: -0.0087919, -0.0074150, -0.0086521, -0.0074062, -0.0013856, 0.0012372
8: 0.0032749, 0.0076348, 0.0032480, 0.0071784, -0.0032933, 0.0037233
9: -0.0047553, -0.0022582, -0.0045018, -0.0022424, -0.0025129, 0.0022436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010108
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010735, upper bound: 0.0010329
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006415, 0.0014257, 0.0006561, 0.0013691, -0.0007181, 0.0007602
1: 0.9928668, 0.9948441, 0.9930096, 0.9948073, -0.0018252, 0.0017186
2: -0.0071263, -0.0045994, -0.0068779, -0.0046340, -0.0024923, 0.0022785
3: 0.0034823, 0.0043024, 0.0034976, 0.0042432, -0.0006820, 0.0007262
4: 0.0023729, 0.0040492, 0.0023898, 0.0038529, -0.0014800, 0.0016594
5: 0.0052633, 0.0071850, 0.0052991, 0.0070462, -0.0017830, 0.0018859
6: -0.0015947, -0.0006976, -0.0015085, -0.0007103, -0.0008844, 0.0008108
7: -0.0087857, -0.0074020, -0.0086858, -0.0074277, -0.0013580, 0.0012838
8: 0.0032347, 0.0076147, 0.0033144, 0.0072883, -0.0033361, 0.0035961
9: -0.0047441, -0.0022346, -0.0045629, -0.0022813, -0.0024628, 0.0023283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010943, upper bound: 0.0010322
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010943, upper bound: 0.0010368
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0006359, 0.0014181, 0.0006561, 0.0013691, -0.0007237, 0.0007527
1: 0.9928859, 0.9948581, 0.9930096, 0.9948073, -0.0018073, 0.0017332
2: -0.0070929, -0.0045860, -0.0068779, -0.0046340, -0.0024590, 0.0022919
3: 0.0034764, 0.0042945, 0.0034976, 0.0042432, -0.0006886, 0.0007190
4: 0.0023663, 0.0040229, 0.0023898, 0.0038529, -0.0014866, 0.0016331
5: 0.0052494, 0.0071664, 0.0052991, 0.0070462, -0.0017968, 0.0018673
6: -0.0015831, -0.0006927, -0.0015085, -0.0007103, -0.0008729, 0.0008157
7: -0.0087723, -0.0073920, -0.0086858, -0.0074277, -0.0013446, 0.0012938
8: 0.0032039, 0.0075709, 0.0033144, 0.0072883, -0.0033824, 0.0035609
9: -0.0047198, -0.0022165, -0.0045629, -0.0022813, -0.0024385, 0.0023463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010943, upper bound: 0.0010322
time: 1.27 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010943, upper bound: 0.0010370
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006440, 0.0013500, -0.0007033, 0.0007726
1: 0.9928666, 0.9948542, 0.9930576, 0.9948378, -0.0018572, 0.0016827
2: -0.0071265, -0.0045897, -0.0067942, -0.0046051, -0.0025213, 0.0022045
3: 0.0034781, 0.0043025, 0.0034849, 0.0042232, -0.0006673, 0.0007396
4: 0.0023681, 0.0040494, 0.0023757, 0.0037868, -0.0014187, 0.0016737
5: 0.0052533, 0.0071851, 0.0052692, 0.0069995, -0.0017462, 0.0019159
6: -0.0015947, -0.0006941, -0.0014794, -0.0006997, -0.0008950, 0.0007853
7: -0.0087858, -0.0073948, -0.0086521, -0.0074062, -0.0013796, 0.0012574
8: 0.0032124, 0.0076149, 0.0032480, 0.0071784, -0.0032985, 0.0036291
9: -0.0047442, -0.0022215, -0.0045018, -0.0022424, -0.0025019, 0.0022803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010943, upper bound: 0.0010377
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010943, upper bound: 0.0010536
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006440, 0.0013500, -0.0007090, 0.0007650
1: 0.9928859, 0.9948689, 0.9930576, 0.9948378, -0.0018391, 0.0016976
2: -0.0070932, -0.0045761, -0.0067942, -0.0046051, -0.0024880, 0.0022180
3: 0.0034720, 0.0042945, 0.0034849, 0.0042232, -0.0006740, 0.0007322
4: 0.0023615, 0.0040230, 0.0023757, 0.0037868, -0.0014253, 0.0016474
5: 0.0052392, 0.0071665, 0.0052692, 0.0069995, -0.0017603, 0.0018973
6: -0.0015832, -0.0006891, -0.0014794, -0.0006997, -0.0008835, 0.0007903
7: -0.0087724, -0.0073846, -0.0086521, -0.0074062, -0.0013662, 0.0012675
8: 0.0031812, 0.0075711, 0.0032480, 0.0071784, -0.0033454, 0.0035917
9: -0.0047199, -0.0022032, -0.0045018, -0.0022424, -0.0024776, 0.0022986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010943, upper bound: 0.0010377
time: 1.16 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010943, upper bound: 0.0010550
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006592, 0.0014364, 0.0006502, 0.0013620, -0.0006937, 0.0007772
1: 0.9928398, 0.9947996, 0.9930273, 0.9948221, -0.0018722, 0.0016597
2: -0.0071732, -0.0046411, -0.0068469, -0.0046198, -0.0025533, 0.0022058
3: 0.0035008, 0.0043136, 0.0034914, 0.0042358, -0.0006584, 0.0007474
4: 0.0023933, 0.0040863, 0.0023829, 0.0038284, -0.0014351, 0.0017034
5: 0.0053064, 0.0072112, 0.0052845, 0.0070289, -0.0017225, 0.0019268
6: -0.0016110, -0.0007129, -0.0014977, -0.0007051, -0.0009058, 0.0007848
7: -0.0088046, -0.0074330, -0.0086733, -0.0074172, -0.0013874, 0.0012403
8: 0.0033308, 0.0076763, 0.0032819, 0.0072476, -0.0032667, 0.0038012
9: -0.0047783, -0.0022910, -0.0045402, -0.0022623, -0.0025161, 0.0022493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010041
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010692, upper bound: 0.0010100
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006551, 0.0014364, 0.0006368, 0.0013433, -0.0006792, 0.0007907
1: 0.9928396, 0.9948098, 0.9930744, 0.9948560, -0.0019063, 0.0016250
2: -0.0071734, -0.0046315, -0.0067649, -0.0045882, -0.0025852, 0.0021334
3: 0.0034965, 0.0043137, 0.0034774, 0.0042163, -0.0006447, 0.0007617
4: 0.0023886, 0.0040865, 0.0023674, 0.0037636, -0.0013750, 0.0017191
5: 0.0052965, 0.0072114, 0.0052517, 0.0069831, -0.0016866, 0.0019597
6: -0.0016110, -0.0007094, -0.0014693, -0.0006935, -0.0009175, 0.0007599
7: -0.0088047, -0.0074259, -0.0086403, -0.0073936, -0.0014111, 0.0012145
8: 0.0033087, 0.0076766, 0.0032089, 0.0071399, -0.0032382, 0.0038357
9: -0.0047785, -0.0022780, -0.0044804, -0.0022194, -0.0025591, 0.0022025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010733, upper bound: 0.0010120
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010733, upper bound: 0.0010120
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006415, 0.0014257, 0.0006502, 0.0013620, -0.0007112, 0.0007662
1: 0.9928668, 0.9948441, 0.9930273, 0.9948221, -0.0018415, 0.0017022
2: -0.0071263, -0.0045994, -0.0068469, -0.0046198, -0.0025064, 0.0022475
3: 0.0034823, 0.0043024, 0.0034914, 0.0042358, -0.0006752, 0.0007340
4: 0.0023729, 0.0040492, 0.0023829, 0.0038284, -0.0014556, 0.0016663
5: 0.0052633, 0.0071850, 0.0052845, 0.0070289, -0.0017657, 0.0019005
6: -0.0015947, -0.0006976, -0.0014977, -0.0007051, -0.0008896, 0.0008001
7: -0.0087857, -0.0074020, -0.0086733, -0.0074172, -0.0013685, 0.0012714
8: 0.0032347, 0.0076147, 0.0032819, 0.0072476, -0.0033223, 0.0036733
9: -0.0047441, -0.0022346, -0.0045402, -0.0022623, -0.0024818, 0.0023057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010888, upper bound: 0.0010350
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010888, upper bound: 0.0010377
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006375, 0.0014257, 0.0006368, 0.0013433, -0.0006968, 0.0007797
1: 0.9928666, 0.9948542, 0.9930744, 0.9948560, -0.0018763, 0.0016676
2: -0.0071265, -0.0045897, -0.0067649, -0.0045882, -0.0025383, 0.0021752
3: 0.0034781, 0.0043025, 0.0034774, 0.0042163, -0.0006613, 0.0007483
4: 0.0023681, 0.0040494, 0.0023674, 0.0037636, -0.0013955, 0.0016820
5: 0.0052533, 0.0071851, 0.0052517, 0.0069831, -0.0017299, 0.0019335
6: -0.0015947, -0.0006941, -0.0014693, -0.0006935, -0.0009012, 0.0007752
7: -0.0087858, -0.0073948, -0.0086403, -0.0073936, -0.0013922, 0.0012456
8: 0.0032124, 0.0076149, 0.0032089, 0.0071399, -0.0032903, 0.0037085
9: -0.0047442, -0.0022215, -0.0044804, -0.0022194, -0.0025248, 0.0022589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010912, upper bound: 0.0010406
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010912, upper bound: 0.0010547
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0006530, 0.0014291, 0.0006502, 0.0013620, -0.0006999, 0.0007701
1: 0.9928580, 0.9948150, 0.9930273, 0.9948221, -0.0018561, 0.0016767
2: -0.0071414, -0.0046266, -0.0068469, -0.0046198, -0.0025215, 0.0022203
3: 0.0034943, 0.0043060, 0.0034914, 0.0042358, -0.0006661, 0.0007408
4: 0.0023862, 0.0040612, 0.0023829, 0.0038284, -0.0014422, 0.0016783
5: 0.0052914, 0.0071935, 0.0052845, 0.0070289, -0.0017375, 0.0019090
6: -0.0015999, -0.0007076, -0.0014977, -0.0007051, -0.0008948, 0.0007901
7: -0.0087918, -0.0074222, -0.0086733, -0.0074172, -0.0013746, 0.0012511
8: 0.0032974, 0.0076345, 0.0032819, 0.0072476, -0.0033272, 0.0037827
9: -0.0047551, -0.0022714, -0.0045402, -0.0022623, -0.0024928, 0.0022689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010722, upper bound: 0.0010032
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010722, upper bound: 0.0010121
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0006489, 0.0014292, 0.0006368, 0.0013433, -0.0006854, 0.0007835
1: 0.9928579, 0.9948256, 0.9930744, 0.9948560, -0.0018893, 0.0016418
2: -0.0071416, -0.0046168, -0.0067649, -0.0045882, -0.0025534, 0.0021481
3: 0.0034900, 0.0043061, 0.0034774, 0.0042163, -0.0006523, 0.0007545
4: 0.0023814, 0.0040613, 0.0023674, 0.0037636, -0.0013822, 0.0016940
5: 0.0052813, 0.0071936, 0.0052517, 0.0069831, -0.0017018, 0.0019419
6: -0.0016000, -0.0007040, -0.0014693, -0.0006935, -0.0009065, 0.0007653
7: -0.0087919, -0.0074150, -0.0086403, -0.0073936, -0.0013983, 0.0012254
8: 0.0032749, 0.0076348, 0.0032089, 0.0071399, -0.0032895, 0.0038208
9: -0.0047553, -0.0022582, -0.0044804, -0.0022194, -0.0025358, 0.0022223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010763, upper bound: 0.0010108
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010763, upper bound: 0.0010346
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0006359, 0.0014181, 0.0006502, 0.0013620, -0.0007169, 0.0007588
1: 0.9928859, 0.9948581, 0.9930273, 0.9948221, -0.0018250, 0.0017181
2: -0.0070929, -0.0045860, -0.0068469, -0.0046198, -0.0024731, 0.0022609
3: 0.0034764, 0.0042945, 0.0034914, 0.0042358, -0.0006825, 0.0007271
4: 0.0023663, 0.0040229, 0.0023829, 0.0038284, -0.0014621, 0.0016400
5: 0.0052494, 0.0071664, 0.0052845, 0.0070289, -0.0017795, 0.0018819
6: -0.0015831, -0.0006927, -0.0014977, -0.0007051, -0.0008780, 0.0008050
7: -0.0087723, -0.0073920, -0.0086733, -0.0074172, -0.0013551, 0.0012813
8: 0.0032039, 0.0075709, 0.0032819, 0.0072476, -0.0033811, 0.0036469
9: -0.0047198, -0.0022165, -0.0045402, -0.0022623, -0.0024575, 0.0023237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010894, upper bound: 0.0010322
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0010894, upper bound: 0.0010370
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006317, 0.0014181, 0.0006368, 0.0013433, -0.0007025, 0.0007723
1: 0.9928859, 0.9948689, 0.9930744, 0.9948560, -0.0018586, 0.0016834
2: -0.0070932, -0.0045761, -0.0067649, -0.0045882, -0.0025050, 0.0021888
3: 0.0034720, 0.0042945, 0.0034774, 0.0042163, -0.0006685, 0.0007412
4: 0.0023615, 0.0040230, 0.0023674, 0.0037636, -0.0014022, 0.0016557
5: 0.0052392, 0.0071665, 0.0052517, 0.0069831, -0.0017439, 0.0019149
6: -0.0015832, -0.0006891, -0.0014693, -0.0006935, -0.0008897, 0.0007801
7: -0.0087724, -0.0073846, -0.0086403, -0.0073936, -0.0013788, 0.0012557
8: 0.0031812, 0.0075711, 0.0032089, 0.0071399, -0.0033414, 0.0036863
9: -0.0047199, -0.0022032, -0.0044804, -0.0022194, -0.0025005, 0.0022773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.02 + 597.35 = 600.37 seconds
