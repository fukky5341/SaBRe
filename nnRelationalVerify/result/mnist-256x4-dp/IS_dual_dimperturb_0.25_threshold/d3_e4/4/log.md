## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00221263


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0055650, 0.0102769, 0.0055650, 0.0102769, -0.0045898, 0.0045898)
1: (0.0010260, 0.0026136, 0.0010260, 0.0026136, -0.0015407, 0.0015407)
2: (0.0083052, 0.0112831, 0.0083052, 0.0112831, -0.0029617, 0.0029617)
3: (-0.0049398, -0.0020031, -0.0049398, -0.0020031, -0.0025135, 0.0025135)
4: (-0.0008702, 0.0013107, -0.0008702, 0.0013107, -0.0015987, 0.0015987)
5: (0.0023910, 0.0048491, 0.0023910, 0.0048491, -0.0022974, 0.0022974)
6: (-0.0113239, -0.0030606, -0.0113239, -0.0030606, -0.0063617, 0.0063617)
7: (0.0005423, 0.0122896, 0.0005423, 0.0122896, -0.0088518, 0.0088518)
8: (0.9903491, 0.9980031, 0.9903491, 0.9980031, -0.0055955, 0.0055955)
9: (-0.0139547, -0.0067792, -0.0139547, -0.0067792, -0.0052774, 0.0052774)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.93 + 1.53 = 3.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0030574, upper bound: 0.0030574

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027250, upper bound: 0.0029007
time: 0.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0029207, upper bound: 0.0029207
time: 0.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.53 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 8, lower bound: -0.0027250, upper bound: 0.0029007
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 8, lower bound: -0.0029207, upper bound: 0.0029207

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0059508, 0.0102289, 0.0057350, 0.0102307, -0.0041672, 0.0044873
1: 0.0009766, 0.0025995, 0.0010447, 0.0026130, -0.0015610, 0.0014783
2: 0.0083123, 0.0110698, 0.0083365, 0.0111891, -0.0028768, 0.0027215
3: -0.0048838, -0.0021149, -0.0049374, -0.0020880, -0.0023082, 0.0023812
4: -0.0006421, 0.0012500, -0.0007665, 0.0013081, -0.0013646, 0.0013792
5: 0.0024384, 0.0046231, 0.0024138, 0.0047495, -0.0021869, 0.0020558
6: -0.0110172, -0.0039573, -0.0112882, -0.0034557, -0.0055669, 0.0054506
7: 0.0016317, 0.0119793, 0.0010548, 0.0122764, -0.0077114, 0.0077421
8: 0.9912093, 0.9977520, 0.9907282, 0.9979856, -0.0047348, 0.0047903
9: -0.0137563, -0.0075199, -0.0139462, -0.0071175, -0.0045603, 0.0045141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025872, upper bound: 0.0027029
time: 0.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025953, upper bound: 0.0027639
time: 0.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0056849, 0.0102308, 0.0056118, 0.0102589, -0.0042510, 0.0045169
1: 0.0010398, 0.0026127, 0.0010317, 0.0026133, -0.0014728, 0.0015249
2: 0.0083342, 0.0112168, 0.0083165, 0.0112572, -0.0029172, 0.0027911
3: -0.0049363, -0.0020846, -0.0049385, -0.0020341, -0.0024739, 0.0022628
4: -0.0007933, 0.0013069, -0.0008398, 0.0013093, -0.0013142, 0.0015706
5: 0.0024184, 0.0047788, 0.0024014, 0.0048217, -0.0022590, 0.0020726
6: -0.0112743, -0.0033393, -0.0113053, -0.0031694, -0.0062445, 0.0053067
7: 0.0009494, 0.0122701, 0.0007023, 0.0122824, -0.0074088, 0.0086972
8: 0.9906164, 0.9979780, 0.9904534, 0.9979938, -0.0045753, 0.0054964
9: -0.0139421, -0.0070323, -0.0139501, -0.0068793, -0.0051832, 0.0043466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027752, upper bound: 0.0027190
time: 0.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027863, upper bound: 0.0027862
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.17 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 8, lower bound: -0.0025872, upper bound: 0.0027029
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 8, lower bound: -0.0025953, upper bound: 0.0027639
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 8, lower bound: -0.0027752, upper bound: 0.0027190
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 8, lower bound: -0.0027863, upper bound: 0.0027862

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0060275, 0.0097972, 0.0059286, 0.0093528, -0.0027073, 0.0036839
1: 0.0012689, 0.0025993, 0.0017264, 0.0026124, -0.0012595, 0.0007566
2: 0.0086705, 0.0110274, 0.0090908, 0.0110821, -0.0024021, 0.0016633
3: -0.0048830, -0.0024123, -0.0049348, -0.0027883, -0.0014777, 0.0020235
4: -0.0005720, 0.0012491, -0.0005939, 0.0013052, -0.0012575, 0.0011451
5: 0.0025795, 0.0045781, 0.0027092, 0.0046361, -0.0018141, 0.0014353
6: -0.0109370, -0.0041357, -0.0110566, -0.0039058, -0.0047659, 0.0046841
7: 0.0021396, 0.0119749, 0.0022852, 0.0122617, -0.0069760, 0.0061309
8: 0.9913805, 0.9977275, 0.9911599, 0.9979053, -0.0043091, 0.0041336
9: -0.0137534, -0.0077603, -0.0139368, -0.0077071, -0.0037643, 0.0041515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025872, upper bound: 0.0025504
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025872, upper bound: 0.0027029
time: 0.74 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060211, 0.0099245, 0.0058829, 0.0097413, -0.0033255, 0.0039143
1: 0.0011751, 0.0025993, 0.0014367, 0.0026274, -0.0013807, 0.0010639
2: 0.0085584, 0.0110309, 0.0087861, 0.0111074, -0.0025489, 0.0021039
3: -0.0048830, -0.0023244, -0.0049942, -0.0024951, -0.0018015, 0.0022332
4: -0.0005827, 0.0012491, -0.0006455, 0.0013695, -0.0013906, 0.0011962
5: 0.0025363, 0.0045819, 0.0025497, 0.0046629, -0.0019185, 0.0017095
6: -0.0109618, -0.0041207, -0.0113763, -0.0037994, -0.0049200, 0.0052751
7: 0.0020406, 0.0119749, 0.0018687, 0.0125907, -0.0077291, 0.0065665
8: 0.9913661, 0.9977343, 0.9910579, 0.9981622, -0.0047883, 0.0042376
9: -0.0137534, -0.0077218, -0.0141472, -0.0075274, -0.0039436, 0.0045918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025953, upper bound: 0.0026128
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025953, upper bound: 0.0027639
time: 0.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0057616, 0.0098148, 0.0058055, 0.0093737, -0.0027023, 0.0037357
1: 0.0013200, 0.0026124, 0.0017080, 0.0026125, -0.0011780, 0.0008123
2: 0.0086787, 0.0111744, 0.0090756, 0.0111501, -0.0023811, 0.0016811
3: -0.0049349, -0.0023652, -0.0049355, -0.0027234, -0.0016838, 0.0018993
4: -0.0007257, 0.0013054, -0.0006681, 0.0013060, -0.0011939, 0.0013765
5: 0.0025450, 0.0047339, 0.0027006, 0.0047082, -0.0018991, 0.0014120
6: -0.0111844, -0.0035175, -0.0110678, -0.0036196, -0.0055417, 0.0044746
7: 0.0014308, 0.0122624, 0.0019140, 0.0122658, -0.0066130, 0.0073162
8: 0.9907874, 0.9979461, 0.9908854, 0.9979108, -0.0041001, 0.0049615
9: -0.0139373, -0.0072628, -0.0139394, -0.0074654, -0.0045212, 0.0039414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027579, upper bound: 0.0025434
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027579, upper bound: 0.0025747
time: 0.69 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0057580, 0.0099228, 0.0057629, 0.0097615, -0.0033544, 0.0039413
1: 0.0012382, 0.0026124, 0.0014220, 0.0026276, -0.0012929, 0.0011173
2: 0.0085851, 0.0111764, 0.0087712, 0.0111737, -0.0025216, 0.0021426
3: -0.0049349, -0.0022934, -0.0049950, -0.0024413, -0.0020036, 0.0021199
4: -0.0007332, 0.0013054, -0.0007176, 0.0013704, -0.0013505, 0.0014323
5: 0.0025104, 0.0047360, 0.0025412, 0.0047331, -0.0019881, 0.0017033
6: -0.0112091, -0.0035093, -0.0113874, -0.0035206, -0.0056940, 0.0051438
7: 0.0013505, 0.0122625, 0.0015154, 0.0125950, -0.0074760, 0.0077679
8: 0.9907795, 0.9979539, 0.9907904, 0.9981678, -0.0046593, 0.0050809
9: -0.0139373, -0.0072356, -0.0141499, -0.0072927, -0.0047161, 0.0044573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027639, upper bound: 0.0025953
time: 0.66 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0027639, upper bound: 0.0026322
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.21 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 8, lower bound: -0.0025872, upper bound: 0.0025504
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 8, lower bound: -0.0025872, upper bound: 0.0027029
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 8, lower bound: -0.0025953, upper bound: 0.0026128
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 8, lower bound: -0.0025953, upper bound: 0.0027639
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 8, lower bound: -0.0027579, upper bound: 0.0025434
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 8, lower bound: -0.0027579, upper bound: 0.0025747
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 8, lower bound: -0.0027639, upper bound: 0.0025953
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 8, lower bound: -0.0027639, upper bound: 0.0026322

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0060275, 0.0097972, 0.0061451, 0.0092850, -0.0026938, 0.0034808
1: 0.0012689, 0.0025993, 0.0016936, 0.0025991, -0.0012312, 0.0007892
2: 0.0086705, 0.0110274, 0.0091062, 0.0109624, -0.0022898, 0.0016935
3: -0.0048830, -0.0024123, -0.0048823, -0.0028516, -0.0014177, 0.0019117
4: -0.0005720, 0.0012491, -0.0004656, 0.0012484, -0.0011365, 0.0010244
5: 0.0025795, 0.0045781, 0.0027656, 0.0045093, -0.0016951, 0.0013894
6: -0.0109370, -0.0041357, -0.0108177, -0.0044089, -0.0042938, 0.0042698
7: 0.0021396, 0.0119749, 0.0029049, 0.0119713, -0.0063571, 0.0055382
8: 0.9913805, 0.9977275, 0.9916426, 0.9976931, -0.0038813, 0.0036807
9: -0.0137534, -0.0077603, -0.0137511, -0.0081240, -0.0033715, 0.0037558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024280, upper bound: 0.0023396
time: 0.76 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025167, upper bound: 0.0024772
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060275, 0.0097972, 0.0058799, 0.0093495, -0.0027136, 0.0038144
1: 0.0012689, 0.0025993, 0.0017232, 0.0026122, -0.0012586, 0.0007585
2: 0.0086705, 0.0110274, 0.0090922, 0.0111090, -0.0024386, 0.0016690
3: -0.0048830, -0.0024123, -0.0049340, -0.0027754, -0.0015190, 0.0020200
4: -0.0005720, 0.0012491, -0.0006219, 0.0013044, -0.0012536, 0.0012249
5: 0.0025795, 0.0045781, 0.0027121, 0.0046646, -0.0018905, 0.0014373
6: -0.0109370, -0.0041357, -0.0110504, -0.0037924, -0.0050693, 0.0046811
7: 0.0021396, 0.0119749, 0.0021548, 0.0122577, -0.0069562, 0.0065177
8: 0.9913805, 0.9977275, 0.9910511, 0.9979013, -0.0043026, 0.0044247
9: -0.0137534, -0.0077603, -0.0139342, -0.0076165, -0.0040241, 0.0041388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024280, upper bound: 0.0024888
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025167, upper bound: 0.0026301
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0060211, 0.0099245, 0.0060956, 0.0097298, -0.0033967, 0.0037125
1: 0.0011751, 0.0025993, 0.0013730, 0.0026133, -0.0013581, 0.0011225
2: 0.0085584, 0.0110309, 0.0087608, 0.0109898, -0.0024314, 0.0021952
3: -0.0048830, -0.0023244, -0.0049385, -0.0025237, -0.0017713, 0.0021438
4: -0.0005827, 0.0012491, -0.0005231, 0.0013092, -0.0012938, 0.0010774
5: 0.0025363, 0.0045819, 0.0025905, 0.0045383, -0.0018003, 0.0017014
6: -0.0109618, -0.0041207, -0.0111168, -0.0042938, -0.0044511, 0.0049397
7: 0.0020406, 0.0119749, 0.0024331, 0.0122823, -0.0072340, 0.0059950
8: 0.9913661, 0.9977343, 0.9915321, 0.9979340, -0.0044444, 0.0037878
9: -0.0137534, -0.0077218, -0.0139500, -0.0079230, -0.0035578, 0.0042752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024386, upper bound: 0.0023975
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025257, upper bound: 0.0025425
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060211, 0.0099245, 0.0058355, 0.0097328, -0.0033264, 0.0040470
1: 0.0011751, 0.0025993, 0.0014347, 0.0026272, -0.0013797, 0.0010621
2: 0.0085584, 0.0110309, 0.0087893, 0.0111336, -0.0025751, 0.0021057
3: -0.0048830, -0.0023244, -0.0049935, -0.0024943, -0.0018239, 0.0022295
4: -0.0005827, 0.0012491, -0.0006714, 0.0013688, -0.0013866, 0.0012744
5: 0.0025363, 0.0045819, 0.0025550, 0.0046906, -0.0019962, 0.0017089
6: -0.0109618, -0.0041207, -0.0113652, -0.0036893, -0.0052285, 0.0052670
7: 0.0020406, 0.0119749, 0.0017571, 0.0125866, -0.0077085, 0.0069234
8: 0.9913661, 0.9977343, 0.9909522, 0.9981565, -0.0047825, 0.0045336
9: -0.0137534, -0.0077218, -0.0141445, -0.0074446, -0.0041972, 0.0045786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024386, upper bound: 0.0025410
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025257, upper bound: 0.0026959
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0057616, 0.0098148, 0.0061451, 0.0092850, -0.0030250, 0.0033844
1: 0.0013200, 0.0026124, 0.0016936, 0.0025991, -0.0011825, 0.0008212
2: 0.0086787, 0.0111744, 0.0091062, 0.0109624, -0.0021869, 0.0018766
3: -0.0049349, -0.0023652, -0.0048823, -0.0028516, -0.0015487, 0.0019757
4: -0.0007257, 0.0013054, -0.0004656, 0.0012484, -0.0013321, 0.0011634
5: 0.0025450, 0.0047339, 0.0027656, 0.0045093, -0.0016933, 0.0015835
6: -0.0111844, -0.0035175, -0.0108177, -0.0044089, -0.0047252, 0.0050397
7: 0.0014308, 0.0122624, 0.0029049, 0.0119713, -0.0072783, 0.0062565
8: 0.9907874, 0.9979461, 0.9916426, 0.9976931, -0.0046199, 0.0041782
9: -0.0139373, -0.0072628, -0.0137511, -0.0081240, -0.0038268, 0.0043899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025314, upper bound: 0.0023840
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026889, upper bound: 0.0024718
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0057616, 0.0098148, 0.0058799, 0.0093495, -0.0026829, 0.0034046
1: 0.0013200, 0.0026124, 0.0017232, 0.0026122, -0.0011774, 0.0007577
2: 0.0086787, 0.0111744, 0.0090922, 0.0111090, -0.0022201, 0.0016677
3: -0.0049349, -0.0023652, -0.0049340, -0.0027754, -0.0014554, 0.0018968
4: -0.0007257, 0.0013054, -0.0006219, 0.0013044, -0.0011912, 0.0011134
5: 0.0025450, 0.0047339, 0.0027121, 0.0046646, -0.0016818, 0.0014031
6: -0.0111844, -0.0035175, -0.0110504, -0.0037924, -0.0045569, 0.0044593
7: 0.0014308, 0.0122624, 0.0021548, 0.0122577, -0.0065991, 0.0059738
8: 0.9907874, 0.9979461, 0.9910511, 0.9979013, -0.0040893, 0.0040035
9: -0.0139373, -0.0072628, -0.0139342, -0.0076165, -0.0036615, 0.0039324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025957, upper bound: 0.0023400
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026889, upper bound: 0.0025049
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0057580, 0.0099228, 0.0060956, 0.0097298, -0.0037019, 0.0035879
1: 0.0012382, 0.0026124, 0.0013730, 0.0026133, -0.0012982, 0.0011552
2: 0.0085851, 0.0111764, 0.0087608, 0.0109898, -0.0023263, 0.0023640
3: -0.0049349, -0.0022934, -0.0049385, -0.0025237, -0.0019055, 0.0021843
4: -0.0007332, 0.0013054, -0.0005231, 0.0013092, -0.0014716, 0.0012215
5: 0.0025104, 0.0047360, 0.0025905, 0.0045383, -0.0017811, 0.0018802
6: -0.0112091, -0.0035093, -0.0111168, -0.0042938, -0.0048727, 0.0056490
7: 0.0013505, 0.0122625, 0.0024331, 0.0122823, -0.0080591, 0.0067459
8: 0.9907795, 0.9979539, 0.9915321, 0.9979340, -0.0051248, 0.0042930
9: -0.0139373, -0.0072356, -0.0139500, -0.0079230, -0.0040314, 0.0048508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025410, upper bound: 0.0024386
time: 0.63 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026959, upper bound: 0.0025257
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0057580, 0.0099228, 0.0058355, 0.0097328, -0.0033331, 0.0036129
1: 0.0012382, 0.0026124, 0.0014347, 0.0026272, -0.0012921, 0.0010627
2: 0.0085851, 0.0111764, 0.0087893, 0.0111336, -0.0023611, 0.0021286
3: -0.0049349, -0.0022934, -0.0049935, -0.0024943, -0.0017825, 0.0021169
4: -0.0007332, 0.0013054, -0.0006714, 0.0013688, -0.0013473, 0.0011686
5: 0.0025104, 0.0047360, 0.0025550, 0.0046906, -0.0017732, 0.0016931
6: -0.0112091, -0.0035093, -0.0113652, -0.0036893, -0.0047089, 0.0051221
7: 0.0013505, 0.0122625, 0.0017571, 0.0125866, -0.0074598, 0.0064323
8: 0.9907795, 0.9979539, 0.9909522, 0.9981565, -0.0046473, 0.0041193
9: -0.0139373, -0.0072356, -0.0141445, -0.0074446, -0.0038552, 0.0044469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026066, upper bound: 0.0024001
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026959, upper bound: 0.0025614
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.35 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0024280, upper bound: 0.0023396
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0025167, upper bound: 0.0024772
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0024280, upper bound: 0.0024888
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0025167, upper bound: 0.0026301
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0024386, upper bound: 0.0023975
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0025257, upper bound: 0.0025425
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0024386, upper bound: 0.0025410
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0025257, upper bound: 0.0026959
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0025314, upper bound: 0.0023840
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0026889, upper bound: 0.0024718
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0025957, upper bound: 0.0023400
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0026889, upper bound: 0.0025049
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0025410, upper bound: 0.0024386
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0026959, upper bound: 0.0025257
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0026066, upper bound: 0.0024001
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 8, lower bound: -0.0026959, upper bound: 0.0025614

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061276, 0.0092832, 0.0063423, 0.0088350, -0.0015492, 0.0023821
1: 0.0016816, 0.0025992, 0.0022386, 0.0025987, -0.0007944, 0.0001970
2: 0.0091138, 0.0109721, 0.0094752, 0.0108534, -0.0015240, 0.0008565
3: -0.0048827, -0.0028323, -0.0048807, -0.0034554, -0.0007796, 0.0014146
4: -0.0004772, 0.0012489, -0.0002963, 0.0012467, -0.0010042, 0.0008440
5: 0.0027633, 0.0045195, 0.0029335, 0.0043937, -0.0012040, 0.0009075
6: -0.0108202, -0.0043682, -0.0106609, -0.0048673, -0.0034939, 0.0036008
7: 0.0028412, 0.0119736, 0.0040721, 0.0119626, -0.0054350, 0.0043157
8: 0.9916036, 0.9976948, 0.9920824, 0.9976406, -0.0034545, 0.0031231
9: -0.0137526, -0.0080857, -0.0137455, -0.0087001, -0.0027596, 0.0033054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023940, upper bound: 0.0023396
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023940, upper bound: 0.0023396
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060521, 0.0096479, 0.0062121, 0.0090370, -0.0021170, 0.0031644
1: 0.0013721, 0.0025992, 0.0019422, 0.0025990, -0.0011259, 0.0005013
2: 0.0087997, 0.0110138, 0.0093185, 0.0109254, -0.0020805, 0.0012703
3: -0.0048828, -0.0025171, -0.0048819, -0.0031187, -0.0010626, 0.0017997
4: -0.0005486, 0.0012490, -0.0004031, 0.0012480, -0.0011162, 0.0008881
5: 0.0026334, 0.0045637, 0.0028513, 0.0044700, -0.0015422, 0.0011518
6: -0.0109080, -0.0041928, -0.0107453, -0.0045646, -0.0038583, 0.0040567
7: 0.0023121, 0.0119741, 0.0033596, 0.0119691, -0.0061943, 0.0047017
8: 0.9914353, 0.9977193, 0.9917920, 0.9976710, -0.0038028, 0.0032732
9: -0.0137529, -0.0078402, -0.0137497, -0.0083386, -0.0029158, 0.0036856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024792, upper bound: 0.0024772
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024792, upper bound: 0.0024772
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0061276, 0.0092832, 0.0060926, 0.0089290, -0.0017515, 0.0027045
1: 0.0016816, 0.0025992, 0.0022025, 0.0026123, -0.0008236, 0.0002436
2: 0.0091138, 0.0109721, 0.0094232, 0.0109914, -0.0017022, 0.0009684
3: -0.0048827, -0.0028323, -0.0049345, -0.0033126, -0.0009640, 0.0015303
4: -0.0004772, 0.0012489, -0.0004509, 0.0013049, -0.0011294, 0.0010436
5: 0.0027633, 0.0045195, 0.0028784, 0.0045400, -0.0013929, 0.0010260
6: -0.0108202, -0.0043682, -0.0108796, -0.0042868, -0.0042433, 0.0040710
7: 0.0028412, 0.0119736, 0.0032815, 0.0122603, -0.0060753, 0.0053364
8: 0.9916036, 0.9976948, 0.9915254, 0.9978504, -0.0039055, 0.0038421
9: -0.0137526, -0.0080857, -0.0139359, -0.0081947, -0.0034122, 0.0037148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023844, upper bound: 0.0024888
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023844, upper bound: 0.0024888
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060521, 0.0096479, 0.0059470, 0.0091044, -0.0021698, 0.0034967
1: 0.0013721, 0.0025992, 0.0019807, 0.0026121, -0.0011532, 0.0004609
2: 0.0087997, 0.0110138, 0.0092937, 0.0110719, -0.0022642, 0.0012653
3: -0.0048828, -0.0025171, -0.0049336, -0.0030464, -0.0011726, 0.0019078
4: -0.0005486, 0.0012490, -0.0005591, 0.0013039, -0.0012333, 0.0010874
5: 0.0026334, 0.0045637, 0.0028046, 0.0046253, -0.0017368, 0.0012148
6: -0.0109080, -0.0041928, -0.0109548, -0.0039484, -0.0046307, 0.0044729
7: 0.0023121, 0.0119741, 0.0026119, 0.0122551, -0.0067931, 0.0056758
8: 0.9914353, 0.9977193, 0.9912009, 0.9978716, -0.0042188, 0.0040142
9: -0.0137529, -0.0078402, -0.0139326, -0.0078321, -0.0035642, 0.0040685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024719, upper bound: 0.0026301
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024719, upper bound: 0.0026301
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061209, 0.0093781, 0.0062979, 0.0089312, -0.0018060, 0.0025999
1: 0.0015950, 0.0025992, 0.0022322, 0.0026126, -0.0009218, 0.0002029
2: 0.0090363, 0.0109758, 0.0094220, 0.0108779, -0.0016810, 0.0009985
3: -0.0048827, -0.0027491, -0.0049357, -0.0034300, -0.0008031, 0.0016520
4: -0.0004874, 0.0012489, -0.0003238, 0.0013063, -0.0011711, 0.0008694
5: 0.0027296, 0.0045235, 0.0028772, 0.0044198, -0.0012994, 0.0010580
6: -0.0108446, -0.0043526, -0.0108845, -0.0047640, -0.0036435, 0.0041977
7: 0.0027480, 0.0119736, 0.0039315, 0.0122671, -0.0063523, 0.0044459
8: 0.9915886, 0.9977018, 0.9919832, 0.9978551, -0.0040271, 0.0032271
9: -0.0137526, -0.0080495, -0.0139402, -0.0086102, -0.0028428, 0.0038554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023584, upper bound: 0.0022922
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023584, upper bound: 0.0022967
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060457, 0.0097629, 0.0061620, 0.0093750, -0.0027678, 0.0034046
1: 0.0012795, 0.0025992, 0.0016536, 0.0026132, -0.0012516, 0.0008131
2: 0.0086938, 0.0110173, 0.0090572, 0.0109531, -0.0022473, 0.0017350
3: -0.0048828, -0.0024304, -0.0049380, -0.0028185, -0.0013896, 0.0020304
4: -0.0005594, 0.0012490, -0.0004589, 0.0013087, -0.0012733, 0.0009432
5: 0.0025917, 0.0045675, 0.0027092, 0.0044994, -0.0016522, 0.0014414
6: -0.0109330, -0.0041779, -0.0110463, -0.0044481, -0.0040467, 0.0047128
7: 0.0022160, 0.0119741, 0.0029137, 0.0122796, -0.0070639, 0.0051429
8: 0.9914210, 0.9977261, 0.9916803, 0.9979103, -0.0043627, 0.0033991
9: -0.0137529, -0.0078020, -0.0139482, -0.0081441, -0.0031076, 0.0042032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024337, upper bound: 0.0024302
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024337, upper bound: 0.0024337
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0061209, 0.0093781, 0.0060512, 0.0090268, -0.0019606, 0.0029223
1: 0.0015950, 0.0025992, 0.0021965, 0.0026264, -0.0009441, 0.0002495
2: 0.0090363, 0.0109758, 0.0093692, 0.0110143, -0.0018593, 0.0010840
3: -0.0048827, -0.0027491, -0.0049904, -0.0032889, -0.0009874, 0.0017404
4: -0.0004874, 0.0012489, -0.0004765, 0.0013655, -0.0012668, 0.0010690
5: 0.0027296, 0.0045235, 0.0028212, 0.0045643, -0.0014883, 0.0011485
6: -0.0108446, -0.0043526, -0.0111068, -0.0041906, -0.0043928, 0.0045570
7: 0.0027480, 0.0119736, 0.0031505, 0.0125698, -0.0068416, 0.0054664
8: 0.9915886, 0.9977018, 0.9914331, 0.9980683, -0.0043718, 0.0039460
9: -0.0137526, -0.0080495, -0.0141338, -0.0081109, -0.0034953, 0.0041682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023407, upper bound: 0.0024466
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023407, upper bound: 0.0024430
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060457, 0.0097629, 0.0059033, 0.0094122, -0.0027274, 0.0037361
1: 0.0012795, 0.0025992, 0.0017198, 0.0026270, -0.0012733, 0.0007444
2: 0.0086938, 0.0110173, 0.0090729, 0.0110961, -0.0024023, 0.0016594
3: -0.0048828, -0.0024304, -0.0049929, -0.0027879, -0.0014461, 0.0021161
4: -0.0005594, 0.0012490, -0.0006072, 0.0013681, -0.0013660, 0.0011389
5: 0.0025917, 0.0045675, 0.0026620, 0.0046509, -0.0018464, 0.0014646
6: -0.0109330, -0.0041779, -0.0112735, -0.0038469, -0.0048172, 0.0050465
7: 0.0022160, 0.0119741, 0.0022313, 0.0125835, -0.0075381, 0.0060805
8: 0.9914210, 0.9977261, 0.9911034, 0.9981266, -0.0046967, 0.0041383
9: -0.0137529, -0.0078020, -0.0141426, -0.0076648, -0.0037432, 0.0045063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024168, upper bound: 0.0025900
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024168, upper bound: 0.0025857
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0059783, 0.0089728, 0.0062417, 0.0089321, -0.0020051, 0.0017076
1: 0.0021374, 0.0026125, 0.0020825, 0.0025990, -0.0003082, 0.0003834
2: 0.0093918, 0.0110546, 0.0094038, 0.0109090, -0.0009585, 0.0011523
3: -0.0049353, -0.0032000, -0.0048821, -0.0032630, -0.0010607, 0.0010775
4: -0.0005264, 0.0013058, -0.0003721, 0.0012482, -0.0011215, 0.0010211
5: 0.0028596, 0.0046070, 0.0028931, 0.0044526, -0.0009874, 0.0011365
6: -0.0109027, -0.0040212, -0.0107074, -0.0046335, -0.0038221, 0.0042637
7: 0.0028649, 0.0122648, 0.0035985, 0.0119700, -0.0057642, 0.0053023
8: 0.9912706, 0.9978600, 0.9918581, 0.9976593, -0.0040459, 0.0036446
9: -0.0139388, -0.0079454, -0.0137503, -0.0084458, -0.0033447, 0.0036694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024899, upper bound: 0.0023840
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024899, upper bound: 0.0023840
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058285, 0.0094580, 0.0061697, 0.0091773, -0.0026969, 0.0027711
1: 0.0015973, 0.0026123, 0.0017891, 0.0025991, -0.0008799, 0.0007183
2: 0.0090067, 0.0111374, 0.0091987, 0.0109488, -0.0017409, 0.0016627
3: -0.0049345, -0.0026412, -0.0048822, -0.0029535, -0.0014412, 0.0016074
4: -0.0006628, 0.0013049, -0.0004422, 0.0012483, -0.0011944, 0.0011434
5: 0.0026660, 0.0046947, 0.0028013, 0.0044949, -0.0014377, 0.0014231
6: -0.0110933, -0.0036730, -0.0107891, -0.0044660, -0.0044599, 0.0045827
7: 0.0018876, 0.0122599, 0.0030765, 0.0119705, -0.0064265, 0.0061016
8: 0.9909366, 0.9979160, 0.9916974, 0.9976845, -0.0042040, 0.0040730
9: -0.0139357, -0.0074786, -0.0137506, -0.0082045, -0.0037574, 0.0039290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026384, upper bound: 0.0024718
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026384, upper bound: 0.0024718
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058647, 0.0093318, 0.0060926, 0.0089290, -0.0016466, 0.0023163
1: 0.0017294, 0.0026123, 0.0022025, 0.0026123, -0.0007350, 0.0002180
2: 0.0091113, 0.0111174, 0.0094232, 0.0109914, -0.0014524, 0.0009104
3: -0.0049346, -0.0027740, -0.0049345, -0.0033126, -0.0008629, 0.0013989
4: -0.0006300, 0.0013051, -0.0004509, 0.0013049, -0.0010595, 0.0009342
5: 0.0027170, 0.0046735, 0.0028784, 0.0045400, -0.0012025, 0.0009646
6: -0.0110474, -0.0037572, -0.0108796, -0.0042868, -0.0037721, 0.0038271
7: 0.0021240, 0.0122609, 0.0032815, 0.0122603, -0.0056790, 0.0047771
8: 0.9910174, 0.9979019, 0.9915254, 0.9978504, -0.0036716, 0.0034440
9: -0.0139363, -0.0075911, -0.0139359, -0.0081947, -0.0030546, 0.0034837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025496, upper bound: 0.0023400
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025496, upper bound: 0.0023400
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0057862, 0.0096702, 0.0059470, 0.0091044, -0.0020831, 0.0030942
1: 0.0014212, 0.0026123, 0.0019807, 0.0026121, -0.0010732, 0.0004515
2: 0.0088067, 0.0111608, 0.0092937, 0.0110719, -0.0020114, 0.0012254
3: -0.0049347, -0.0024660, -0.0049336, -0.0030464, -0.0010865, 0.0017856
4: -0.0007028, 0.0013052, -0.0005591, 0.0013039, -0.0011703, 0.0009756
5: 0.0025928, 0.0047195, 0.0028046, 0.0046253, -0.0015347, 0.0011615
6: -0.0111514, -0.0035747, -0.0109548, -0.0039484, -0.0041296, 0.0042453
7: 0.0015966, 0.0122615, 0.0026119, 0.0122551, -0.0064358, 0.0051091
8: 0.9908423, 0.9979352, 0.9912009, 0.9978716, -0.0040050, 0.0035952
9: -0.0139367, -0.0073412, -0.0139326, -0.0078321, -0.0031990, 0.0038600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026548, upper bound: 0.0025049
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026548, upper bound: 0.0025050
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0059747, 0.0090154, 0.0061934, 0.0092548, -0.0026647, 0.0018374
1: 0.0020844, 0.0026125, 0.0017756, 0.0026132, -0.0003972, 0.0007161
2: 0.0093610, 0.0110566, 0.0091585, 0.0109357, -0.0010486, 0.0016268
3: -0.0049353, -0.0031524, -0.0049382, -0.0029507, -0.0013905, 0.0012508
4: -0.0005331, 0.0013058, -0.0004286, 0.0013089, -0.0012610, 0.0010725
5: 0.0028414, 0.0046091, 0.0027511, 0.0044810, -0.0010469, 0.0014309
6: -0.0109222, -0.0040129, -0.0110142, -0.0045211, -0.0039549, 0.0048999
7: 0.0028016, 0.0122648, 0.0031384, 0.0122806, -0.0065066, 0.0057328
8: 0.9912627, 0.9978663, 0.9917502, 0.9979010, -0.0045641, 0.0037500
9: -0.0139388, -0.0079213, -0.0139489, -0.0082484, -0.0035251, 0.0041275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024888, upper bound: 0.0024280
time: 0.70 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024888, upper bound: 0.0023988
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058253, 0.0095264, 0.0061200, 0.0095873, -0.0033814, 0.0029481
1: 0.0015251, 0.0026123, 0.0014777, 0.0026132, -0.0009891, 0.0010469
2: 0.0089394, 0.0111392, 0.0088801, 0.0109763, -0.0018668, 0.0021537
3: -0.0049345, -0.0025790, -0.0049383, -0.0026325, -0.0017897, 0.0018139
4: -0.0006700, 0.0013049, -0.0004996, 0.0013090, -0.0013434, 0.0012010
5: 0.0026404, 0.0046966, 0.0026390, 0.0045240, -0.0015126, 0.0017239
6: -0.0111161, -0.0036655, -0.0110910, -0.0043506, -0.0045969, 0.0052083
7: 0.0018171, 0.0122600, 0.0026097, 0.0122813, -0.0072344, 0.0065829
8: 0.9909294, 0.9979233, 0.9915867, 0.9979254, -0.0047437, 0.0041853
9: -0.0139357, -0.0074525, -0.0139493, -0.0080038, -0.0039598, 0.0044194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025857, upper bound: 0.0024127
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025857, upper bound: 0.0024168
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058619, 0.0094201, 0.0060512, 0.0090268, -0.0019020, 0.0025297
1: 0.0016456, 0.0026123, 0.0021965, 0.0026264, -0.0008620, 0.0002245
2: 0.0090402, 0.0111190, 0.0093692, 0.0110143, -0.0016055, 0.0010516
3: -0.0049346, -0.0026982, -0.0049904, -0.0032889, -0.0008887, 0.0016350
4: -0.0006376, 0.0013051, -0.0004765, 0.0013655, -0.0012257, 0.0009621
5: 0.0026851, 0.0046752, 0.0028212, 0.0045643, -0.0012951, 0.0011142
6: -0.0110767, -0.0037507, -0.0111068, -0.0041906, -0.0039274, 0.0044208
7: 0.0020412, 0.0122610, 0.0031505, 0.0125698, -0.0065916, 0.0049196
8: 0.9910112, 0.9979112, 0.9914331, 0.9980683, -0.0042411, 0.0035582
9: -0.0139364, -0.0075631, -0.0141338, -0.0081109, -0.0031458, 0.0040313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025210, upper bound: 0.0023010
time: 0.78 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025210, upper bound: 0.0023033
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0057824, 0.0097623, 0.0059033, 0.0094122, -0.0026897, 0.0033047
1: 0.0013432, 0.0026123, 0.0017198, 0.0026270, -0.0011841, 0.0007411
2: 0.0087210, 0.0111629, 0.0090729, 0.0110961, -0.0021564, 0.0016544
3: -0.0049347, -0.0023986, -0.0049929, -0.0027879, -0.0013920, 0.0020020
4: -0.0007104, 0.0013052, -0.0006072, 0.0013681, -0.0013262, 0.0010310
5: 0.0025617, 0.0047217, 0.0026620, 0.0046509, -0.0016284, 0.0014332
6: -0.0111754, -0.0035658, -0.0112735, -0.0038469, -0.0042965, 0.0048952
7: 0.0015196, 0.0122616, 0.0022313, 0.0125835, -0.0072865, 0.0055574
8: 0.9908339, 0.9979427, 0.9911034, 0.9981266, -0.0045645, 0.0037201
9: -0.0139367, -0.0073141, -0.0141426, -0.0076648, -0.0033919, 0.0043728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026068, upper bound: 0.0024508
time: 0.70 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026068, upper bound: 0.0024533
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.20 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0023940, upper bound: 0.0023396
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0023940, upper bound: 0.0023396
IS_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024792, upper bound: 0.0024772
IS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024792, upper bound: 0.0024772
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0023844, upper bound: 0.0024888
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0023844, upper bound: 0.0024888
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024719, upper bound: 0.0026301
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024719, upper bound: 0.0026301
IS_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0023584, upper bound: 0.0022922
IS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0023584, upper bound: 0.0022967
IS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024337, upper bound: 0.0024302
IS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024337, upper bound: 0.0024337
IS_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0023407, upper bound: 0.0024466
IS_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0023407, upper bound: 0.0024430
IS_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024168, upper bound: 0.0025900
IS_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024168, upper bound: 0.0025857
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024899, upper bound: 0.0023840
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024899, upper bound: 0.0023840
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0026384, upper bound: 0.0024718
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0026384, upper bound: 0.0024718
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0025496, upper bound: 0.0023400
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0025496, upper bound: 0.0023400
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0026548, upper bound: 0.0025049
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0026548, upper bound: 0.0025050
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024888, upper bound: 0.0024280
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0024888, upper bound: 0.0023988
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0025857, upper bound: 0.0024127
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0025857, upper bound: 0.0024168
IS_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0025210, upper bound: 0.0023010
IS_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0025210, upper bound: 0.0023033
IS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0026068, upper bound: 0.0024508
IS_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 8, lower bound: -0.0026068, upper bound: 0.0024533

## BFS IS instance: IS_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062417, 0.0089321, 0.0063423, 0.0088350, -0.0014256, 0.0015630
1: 0.0020825, 0.0025990, 0.0022386, 0.0025987, -0.0003539, 0.0001968
2: 0.0094038, 0.0109090, 0.0094752, 0.0108534, -0.0009079, 0.0007882
3: -0.0048821, -0.0032630, -0.0048807, -0.0034554, -0.0007790, 0.0009442
4: -0.0003721, 0.0012482, -0.0002963, 0.0012467, -0.0008950, 0.0008433
5: 0.0028931, 0.0044526, 0.0029335, 0.0043937, -0.0008775, 0.0008351
6: -0.0107074, -0.0046335, -0.0106609, -0.0048673, -0.0032362, 0.0033135
7: 0.0035985, 0.0119700, 0.0040721, 0.0119626, -0.0046576, 0.0043125
8: 0.9918581, 0.9976593, 0.9920824, 0.9976406, -0.0031788, 0.0030602
9: -0.0137503, -0.0084458, -0.0137455, -0.0087001, -0.0027575, 0.0029325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022863, upper bound: 0.0022333
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022810, upper bound: 0.0022333
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061934, 0.0092548, 0.0063423, 0.0088350, -0.0015830, 0.0021858
1: 0.0017756, 0.0026132, 0.0022386, 0.0025987, -0.0007016, 0.0002230
2: 0.0091585, 0.0109357, 0.0094752, 0.0108534, -0.0013621, 0.0008752
3: -0.0049382, -0.0029507, -0.0048807, -0.0034554, -0.0008825, 0.0013480
4: -0.0004286, 0.0013089, -0.0002963, 0.0012467, -0.0010195, 0.0009554
5: 0.0027511, 0.0044810, 0.0029335, 0.0043937, -0.0011504, 0.0009273
6: -0.0110142, -0.0045211, -0.0106609, -0.0048673, -0.0037869, 0.0036794
7: 0.0031384, 0.0122806, 0.0040721, 0.0119626, -0.0054805, 0.0048855
8: 0.9917502, 0.9979010, 0.9920824, 0.9976406, -0.0035298, 0.0034963
9: -0.0139489, -0.0082484, -0.0137455, -0.0087001, -0.0031239, 0.0033533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022863, upper bound: 0.0022333
time: 0.78 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022810, upper bound: 0.0022333
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061697, 0.0091773, 0.0062121, 0.0090370, -0.0019933, 0.0022410
1: 0.0017891, 0.0025991, 0.0019422, 0.0025990, -0.0006858, 0.0005012
2: 0.0091987, 0.0109488, 0.0093185, 0.0109254, -0.0014106, 0.0012020
3: -0.0048822, -0.0029535, -0.0048819, -0.0031187, -0.0010620, 0.0013097
4: -0.0004422, 0.0012483, -0.0004031, 0.0012480, -0.0010047, 0.0008875
5: 0.0028013, 0.0044949, 0.0028513, 0.0044700, -0.0011561, 0.0010793
6: -0.0107891, -0.0044660, -0.0107453, -0.0045646, -0.0035230, 0.0037692
7: 0.0030765, 0.0119705, 0.0033596, 0.0119691, -0.0053815, 0.0046984
8: 0.9916974, 0.9976845, 0.9917920, 0.9976710, -0.0035269, 0.0031873
9: -0.0137506, -0.0082045, -0.0137497, -0.0083386, -0.0029137, 0.0033033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023563, upper bound: 0.0023647
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023563, upper bound: 0.0023540
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061200, 0.0095873, 0.0062121, 0.0090370, -0.0021450, 0.0028905
1: 0.0014777, 0.0026132, 0.0019422, 0.0025990, -0.0010286, 0.0005296
2: 0.0088801, 0.0109763, 0.0093185, 0.0109254, -0.0018823, 0.0012858
3: -0.0049383, -0.0026325, -0.0048819, -0.0031187, -0.0011745, 0.0017249
4: -0.0004996, 0.0013090, -0.0004031, 0.0012480, -0.0011281, 0.0010093
5: 0.0026390, 0.0045240, 0.0028513, 0.0044700, -0.0014363, 0.0011681
6: -0.0110910, -0.0043506, -0.0107453, -0.0045646, -0.0040673, 0.0041217
7: 0.0026097, 0.0122813, 0.0033596, 0.0119691, -0.0062241, 0.0053214
8: 0.9915867, 0.9979254, 0.9917920, 0.9976710, -0.0038651, 0.0036491
9: -0.0139493, -0.0080038, -0.0137497, -0.0083386, -0.0033121, 0.0037221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023563, upper bound: 0.0023647
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023563, upper bound: 0.0023540
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062417, 0.0089321, 0.0060926, 0.0089290, -0.0016279, 0.0018855
1: 0.0020825, 0.0025990, 0.0022025, 0.0026123, -0.0003832, 0.0002434
2: 0.0094038, 0.0109090, 0.0094232, 0.0109914, -0.0010862, 0.0009000
3: -0.0048821, -0.0032630, -0.0049345, -0.0033126, -0.0009634, 0.0010599
4: -0.0003721, 0.0012482, -0.0004509, 0.0013049, -0.0010202, 0.0010429
5: 0.0028931, 0.0044526, 0.0028784, 0.0045400, -0.0010664, 0.0009536
6: -0.0107074, -0.0046335, -0.0108796, -0.0042868, -0.0039856, 0.0037836
7: 0.0035985, 0.0119700, 0.0032815, 0.0122603, -0.0052979, 0.0053332
8: 0.9918581, 0.9976593, 0.9915254, 0.9978504, -0.0036299, 0.0037792
9: -0.0137503, -0.0084458, -0.0139359, -0.0081947, -0.0034102, 0.0033420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022706, upper bound: 0.0023849
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022705, upper bound: 0.0023793
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061934, 0.0092548, 0.0060926, 0.0089290, -0.0017853, 0.0025083
1: 0.0017756, 0.0026132, 0.0022025, 0.0026123, -0.0007309, 0.0002696
2: 0.0091585, 0.0109357, 0.0094232, 0.0109914, -0.0015404, 0.0009870
3: -0.0049382, -0.0029507, -0.0049345, -0.0033126, -0.0010669, 0.0014637
4: -0.0004286, 0.0013089, -0.0004509, 0.0013049, -0.0011447, 0.0011550
5: 0.0027511, 0.0044810, 0.0028784, 0.0045400, -0.0013393, 0.0010458
6: -0.0110142, -0.0045211, -0.0108796, -0.0042868, -0.0045364, 0.0041495
7: 0.0031384, 0.0122806, 0.0032815, 0.0122603, -0.0061208, 0.0059062
8: 0.9917502, 0.9979010, 0.9915254, 0.9978504, -0.0039809, 0.0042153
9: -0.0139489, -0.0082484, -0.0139359, -0.0081947, -0.0037766, 0.0037627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022706, upper bound: 0.0023849
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022705, upper bound: 0.0023793
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061697, 0.0091773, 0.0059470, 0.0091044, -0.0020461, 0.0025733
1: 0.0017891, 0.0025991, 0.0019807, 0.0026121, -0.0007132, 0.0004608
2: 0.0091987, 0.0109488, 0.0092937, 0.0110719, -0.0015943, 0.0011969
3: -0.0048822, -0.0029535, -0.0049336, -0.0030464, -0.0011720, 0.0014179
4: -0.0004422, 0.0012483, -0.0005591, 0.0013039, -0.0011218, 0.0010868
5: 0.0028013, 0.0044949, 0.0028046, 0.0046253, -0.0013507, 0.0011423
6: -0.0107891, -0.0044660, -0.0109548, -0.0039484, -0.0042954, 0.0041854
7: 0.0030765, 0.0119705, 0.0026119, 0.0122551, -0.0059803, 0.0056725
8: 0.9916974, 0.9976845, 0.9912009, 0.9978716, -0.0039429, 0.0039283
9: -0.0137506, -0.0082045, -0.0139326, -0.0078321, -0.0035621, 0.0036862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023494, upper bound: 0.0025104
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023494, upper bound: 0.0025083
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061200, 0.0095873, 0.0059470, 0.0091044, -0.0021978, 0.0032228
1: 0.0014777, 0.0026132, 0.0019807, 0.0026121, -0.0010560, 0.0004892
2: 0.0088801, 0.0109763, 0.0092937, 0.0110719, -0.0020660, 0.0012807
3: -0.0049383, -0.0026325, -0.0049336, -0.0030464, -0.0012846, 0.0018331
4: -0.0004996, 0.0013090, -0.0005591, 0.0013039, -0.0012452, 0.0012086
5: 0.0026390, 0.0045240, 0.0028046, 0.0046253, -0.0016310, 0.0012312
6: -0.0110910, -0.0043506, -0.0109548, -0.0039484, -0.0048397, 0.0045379
7: 0.0026097, 0.0122813, 0.0026119, 0.0122551, -0.0068228, 0.0062955
8: 0.9915867, 0.9979254, 0.9912009, 0.9978716, -0.0042811, 0.0043901
9: -0.0139493, -0.0080038, -0.0139326, -0.0078321, -0.0039605, 0.0041050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023494, upper bound: 0.0025104
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023494, upper bound: 0.0025083
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061351, 0.0093585, 0.0063555, 0.0089303, -0.0017826, 0.0025041
1: 0.0016137, 0.0025992, 0.0022405, 0.0026125, -0.0009015, 0.0001932
2: 0.0090527, 0.0109679, 0.0094225, 0.0108461, -0.0016206, 0.0009856
3: -0.0048826, -0.0027725, -0.0049353, -0.0034630, -0.0007648, 0.0016230
4: -0.0004770, 0.0012488, -0.0002881, 0.0013057, -0.0011555, 0.0008279
5: 0.0027370, 0.0045151, 0.0028777, 0.0043860, -0.0012505, 0.0010442
6: -0.0108390, -0.0043857, -0.0108825, -0.0048980, -0.0035019, 0.0041433
7: 0.0028099, 0.0119731, 0.0041139, 0.0122644, -0.0062615, 0.0042336
8: 0.9916203, 0.9976999, 0.9921117, 0.9978532, -0.0039749, 0.0030937
9: -0.0137523, -0.0080839, -0.0139385, -0.0087269, -0.0027071, 0.0038034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022797, upper bound: 0.0022043
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022797, upper bound: 0.0022108
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0061707, 0.0092918, 0.0064079, 0.0089952, -0.0018573, 0.0023918
1: 0.0016761, 0.0025991, 0.0022481, 0.0026219, -0.0008496, 0.0001958
2: 0.0091070, 0.0109482, 0.0093866, 0.0108171, -0.0015346, 0.0010269
3: -0.0048824, -0.0028450, -0.0049724, -0.0034929, -0.0007750, 0.0016028
4: -0.0004510, 0.0012486, -0.0002557, 0.0013459, -0.0011970, 0.0008390
5: 0.0027606, 0.0044943, 0.0028396, 0.0043553, -0.0012071, 0.0010880
6: -0.0108209, -0.0044684, -0.0110334, -0.0050196, -0.0035004, 0.0043170
7: 0.0029712, 0.0119719, 0.0042796, 0.0124699, -0.0064414, 0.0042904
8: 0.9916997, 0.9976942, 0.9922286, 0.9979979, -0.0041415, 0.0031223
9: -0.0137515, -0.0081714, -0.0140699, -0.0088328, -0.0027434, 0.0039369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022797, upper bound: 0.0022090
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022797, upper bound: 0.0022148
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0060600, 0.0097397, 0.0062213, 0.0093037, -0.0026016, 0.0033018
1: 0.0012983, 0.0025992, 0.0017290, 0.0026131, -0.0012317, 0.0007318
2: 0.0087138, 0.0110094, 0.0091207, 0.0109203, -0.0021842, 0.0016133
3: -0.0048827, -0.0024532, -0.0049375, -0.0029167, -0.0012835, 0.0020020
4: -0.0005492, 0.0012489, -0.0004154, 0.0013082, -0.0012579, 0.0008980
5: 0.0026002, 0.0045591, 0.0027347, 0.0044646, -0.0015986, 0.0013741
6: -0.0109274, -0.0042112, -0.0110254, -0.0045861, -0.0038822, 0.0046217
7: 0.0022772, 0.0119736, 0.0031741, 0.0122771, -0.0069756, 0.0048766
8: 0.9914529, 0.9977243, 0.9918126, 0.9979025, -0.0043006, 0.0032524
9: -0.0137526, -0.0078359, -0.0139466, -0.0082890, -0.0029576, 0.0041520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023560, upper bound: 0.0023408
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023560, upper bound: 0.0023537
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060943, 0.0096597, 0.0062647, 0.0092597, -0.0024707, 0.0031889
1: 0.0013639, 0.0025992, 0.0018362, 0.0026224, -0.0011783, 0.0006246
2: 0.0087847, 0.0109905, 0.0091743, 0.0108963, -0.0020978, 0.0014918
3: -0.0048825, -0.0025275, -0.0049743, -0.0030391, -0.0011891, 0.0019786
4: -0.0005231, 0.0012487, -0.0003804, 0.0013481, -0.0012983, 0.0009045
5: 0.0026294, 0.0045390, 0.0027361, 0.0044392, -0.0015550, 0.0013429
6: -0.0109088, -0.0042908, -0.0111353, -0.0046869, -0.0038852, 0.0047529
7: 0.0024388, 0.0119724, 0.0034098, 0.0124808, -0.0071444, 0.0048483
8: 0.9915293, 0.9977186, 0.9919093, 0.9980330, -0.0044535, 0.0032963
9: -0.0137518, -0.0079235, -0.0140769, -0.0084078, -0.0029741, 0.0042816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023560, upper bound: 0.0023421
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023560, upper bound: 0.0023560
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061351, 0.0093585, 0.0061085, 0.0090259, -0.0019372, 0.0028177
1: 0.0016137, 0.0025992, 0.0022048, 0.0026263, -0.0009238, 0.0002398
2: 0.0090527, 0.0109679, 0.0093697, 0.0109826, -0.0017945, 0.0010710
3: -0.0048826, -0.0027725, -0.0049899, -0.0033217, -0.0009492, 0.0017115
4: -0.0004770, 0.0012488, -0.0004410, 0.0013649, -0.0012513, 0.0010276
5: 0.0027370, 0.0045151, 0.0028217, 0.0045307, -0.0014337, 0.0011348
6: -0.0108390, -0.0043857, -0.0111046, -0.0043239, -0.0042239, 0.0045027
7: 0.0028099, 0.0119731, 0.0033321, 0.0125668, -0.0067510, 0.0052549
8: 0.9916203, 0.9976999, 0.9915611, 0.9980661, -0.0043197, 0.0037933
9: -0.0137523, -0.0080839, -0.0141319, -0.0082270, -0.0033601, 0.0041164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022628, upper bound: 0.0023594
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022628, upper bound: 0.0023702
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0061707, 0.0092918, 0.0061687, 0.0090854, -0.0020039, 0.0027042
1: 0.0016761, 0.0025991, 0.0022135, 0.0026349, -0.0008708, 0.0002424
2: 0.0091070, 0.0109482, 0.0093368, 0.0109494, -0.0017085, 0.0011079
3: -0.0048824, -0.0028450, -0.0050239, -0.0033561, -0.0009596, 0.0016866
4: -0.0004510, 0.0012486, -0.0004038, 0.0014017, -0.0012877, 0.0010388
5: 0.0027606, 0.0044943, 0.0027869, 0.0044955, -0.0013904, 0.0011739
6: -0.0108209, -0.0044684, -0.0112429, -0.0044636, -0.0042219, 0.0046577
7: 0.0029712, 0.0119719, 0.0035224, 0.0127551, -0.0069054, 0.0053122
8: 0.9916997, 0.9976942, 0.9916951, 0.9981989, -0.0044684, 0.0038233
9: -0.0137515, -0.0081714, -0.0142523, -0.0083487, -0.0033967, 0.0042336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022628, upper bound: 0.0023553
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022628, upper bound: 0.0023655
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0060600, 0.0097397, 0.0059598, 0.0093507, -0.0025778, 0.0036329
1: 0.0012983, 0.0025992, 0.0017950, 0.0026269, -0.0012533, 0.0006626
2: 0.0087138, 0.0110094, 0.0091258, 0.0110649, -0.0023510, 0.0015508
3: -0.0048827, -0.0024532, -0.0049924, -0.0028846, -0.0013460, 0.0020874
4: -0.0005492, 0.0012489, -0.0005668, 0.0013676, -0.0013503, 0.0010940
5: 0.0026002, 0.0045591, 0.0026868, 0.0046178, -0.0017926, 0.0014028
6: -0.0109274, -0.0042112, -0.0112480, -0.0039781, -0.0046518, 0.0049554
7: 0.0022772, 0.0119736, 0.0024743, 0.0125807, -0.0074484, 0.0058162
8: 0.9914529, 0.9977243, 0.9912293, 0.9981176, -0.0046324, 0.0039907
9: -0.0137526, -0.0078359, -0.0141408, -0.0077995, -0.0035938, 0.0044543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023377, upper bound: 0.0024979
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023377, upper bound: 0.0025161
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060943, 0.0096597, 0.0060137, 0.0093266, -0.0024801, 0.0035177
1: 0.0013639, 0.0025992, 0.0018958, 0.0026354, -0.0011979, 0.0005645
2: 0.0087847, 0.0109905, 0.0091582, 0.0110350, -0.0022504, 0.0014589
3: -0.0048825, -0.0025275, -0.0050261, -0.0029976, -0.0012677, 0.0020565
4: -0.0005231, 0.0012487, -0.0005256, 0.0014041, -0.0013826, 0.0011007
5: 0.0026294, 0.0045390, 0.0026856, 0.0045862, -0.0017476, 0.0013826
6: -0.0109088, -0.0042908, -0.0113558, -0.0041035, -0.0046495, 0.0050534
7: 0.0024388, 0.0119724, 0.0027330, 0.0127676, -0.0075753, 0.0057992
8: 0.9915293, 0.9977186, 0.9913496, 0.9982405, -0.0047582, 0.0040295
9: -0.0137518, -0.0079235, -0.0142603, -0.0079379, -0.0036119, 0.0045571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023377, upper bound: 0.0024952
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023377, upper bound: 0.0025093
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0060926, 0.0089290, 0.0062417, 0.0089321, -0.0018855, 0.0016279
1: 0.0022025, 0.0026123, 0.0020825, 0.0025990, -0.0002434, 0.0003832
2: 0.0094232, 0.0109914, 0.0094038, 0.0109090, -0.0009000, 0.0010862
3: -0.0049345, -0.0033126, -0.0048821, -0.0032630, -0.0010599, 0.0009634
4: -0.0004509, 0.0013049, -0.0003721, 0.0012482, -0.0010429, 0.0010202
5: 0.0028784, 0.0045400, 0.0028931, 0.0044526, -0.0009536, 0.0010664
6: -0.0108796, -0.0042868, -0.0107074, -0.0046335, -0.0037836, 0.0039856
7: 0.0032815, 0.0122603, 0.0035985, 0.0119700, -0.0053332, 0.0052979
8: 0.9915254, 0.9978504, 0.9918581, 0.9976593, -0.0037792, 0.0036299
9: -0.0139359, -0.0081947, -0.0137503, -0.0084458, -0.0033420, 0.0034102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023861, upper bound: 0.0022702
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023795, upper bound: 0.0022702
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0060512, 0.0090268, 0.0062417, 0.0089321, -0.0020029, 0.0017773
1: 0.0021965, 0.0026264, 0.0020825, 0.0025990, -0.0002604, 0.0004047
2: 0.0093692, 0.0110143, 0.0094038, 0.0109090, -0.0009826, 0.0011511
3: -0.0049904, -0.0032889, -0.0048821, -0.0032630, -0.0011453, 0.0010305
4: -0.0004765, 0.0013655, -0.0003721, 0.0012482, -0.0011156, 0.0011127
5: 0.0028212, 0.0045643, 0.0028931, 0.0044526, -0.0010411, 0.0011352
6: -0.0111068, -0.0041906, -0.0107074, -0.0046335, -0.0041309, 0.0042586
7: 0.0031505, 0.0125698, 0.0035985, 0.0119700, -0.0057050, 0.0057709
8: 0.9914331, 0.9980683, 0.9918581, 0.9976593, -0.0040410, 0.0039631
9: -0.0141338, -0.0081109, -0.0137503, -0.0084458, -0.0036444, 0.0036479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023861, upper bound: 0.0022702
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023795, upper bound: 0.0022702
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0059470, 0.0091044, 0.0061697, 0.0091773, -0.0025733, 0.0020461
1: 0.0019807, 0.0026121, 0.0017891, 0.0025991, -0.0004608, 0.0007132
2: 0.0092937, 0.0110719, 0.0091987, 0.0109488, -0.0011969, 0.0015943
3: -0.0049336, -0.0030464, -0.0048822, -0.0029535, -0.0014179, 0.0011720
4: -0.0005591, 0.0013039, -0.0004422, 0.0012483, -0.0010868, 0.0011218
5: 0.0028046, 0.0046253, 0.0028013, 0.0044949, -0.0011423, 0.0013507
6: -0.0109548, -0.0039484, -0.0107891, -0.0044660, -0.0041854, 0.0042954
7: 0.0026119, 0.0122551, 0.0030765, 0.0119705, -0.0056725, 0.0059803
8: 0.9912009, 0.9978716, 0.9916974, 0.9976845, -0.0039283, 0.0039429
9: -0.0139326, -0.0078321, -0.0137506, -0.0082045, -0.0036862, 0.0035621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025177, upper bound: 0.0023491
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025156, upper bound: 0.0023492
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0059033, 0.0094122, 0.0061697, 0.0091773, -0.0026900, 0.0025459
1: 0.0017198, 0.0026270, 0.0017891, 0.0025991, -0.0007574, 0.0007336
2: 0.0090729, 0.0110961, 0.0091987, 0.0109488, -0.0015590, 0.0016589
3: -0.0049929, -0.0027879, -0.0048822, -0.0029535, -0.0014989, 0.0015025
4: -0.0006072, 0.0013681, -0.0004422, 0.0012483, -0.0011827, 0.0012095
5: 0.0026620, 0.0046509, 0.0028013, 0.0044949, -0.0013583, 0.0014191
6: -0.0112735, -0.0038469, -0.0107891, -0.0044660, -0.0046245, 0.0045666
7: 0.0022313, 0.0125835, 0.0030765, 0.0119705, -0.0063172, 0.0064286
8: 0.9911034, 0.9981266, 0.9916974, 0.9976845, -0.0041885, 0.0042919
9: -0.0141426, -0.0076648, -0.0137506, -0.0082045, -0.0039728, 0.0038872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025177, upper bound: 0.0023491
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025156, upper bound: 0.0023491
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0059820, 0.0090152, 0.0060926, 0.0089290, -0.0015209, 0.0015909
1: 0.0020876, 0.0026121, 0.0022025, 0.0026123, -0.0003285, 0.0002094
2: 0.0093605, 0.0110526, 0.0094232, 0.0109914, -0.0009098, 0.0008409
3: -0.0049338, -0.0031576, -0.0049345, -0.0033126, -0.0008286, 0.0009669
4: -0.0005286, 0.0013041, -0.0004509, 0.0013049, -0.0009509, 0.0008970
5: 0.0028422, 0.0046048, 0.0028784, 0.0045400, -0.0009049, 0.0008909
6: -0.0109155, -0.0040299, -0.0108796, -0.0042868, -0.0034211, 0.0035350
7: 0.0028250, 0.0122561, 0.0032815, 0.0122603, -0.0049240, 0.0045870
8: 0.9912789, 0.9978600, 0.9915254, 0.9978504, -0.0033913, 0.0032485
9: -0.0139333, -0.0079361, -0.0139359, -0.0081947, -0.0029331, 0.0031139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024398, upper bound: 0.0022328
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024409, upper bound: 0.0022328
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0059401, 0.0093225, 0.0060926, 0.0089290, -0.0016780, 0.0021420
1: 0.0018253, 0.0026271, 0.0022025, 0.0026123, -0.0006373, 0.0002355
2: 0.0091476, 0.0110757, 0.0094232, 0.0109914, -0.0013062, 0.0009277
3: -0.0049931, -0.0029021, -0.0049345, -0.0033126, -0.0009322, 0.0013289
4: -0.0005756, 0.0013684, -0.0004509, 0.0013049, -0.0010725, 0.0010091
5: 0.0026980, 0.0046293, 0.0028784, 0.0045400, -0.0011524, 0.0009830
6: -0.0112398, -0.0039325, -0.0108796, -0.0042868, -0.0039541, 0.0039001
7: 0.0024476, 0.0125847, 0.0032815, 0.0122603, -0.0057075, 0.0051603
8: 0.9911855, 0.9981169, 0.9915254, 0.9978504, -0.0037416, 0.0036818
9: -0.0141434, -0.0077724, -0.0139359, -0.0081947, -0.0032996, 0.0035234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024409, upper bound: 0.0022349
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024409, upper bound: 0.0022328
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0059047, 0.0092503, 0.0059470, 0.0091044, -0.0019606, 0.0022210
1: 0.0018172, 0.0026121, 0.0019807, 0.0026121, -0.0006460, 0.0004431
2: 0.0091755, 0.0110953, 0.0092937, 0.0110719, -0.0013768, 0.0011577
3: -0.0049339, -0.0028752, -0.0049336, -0.0030464, -0.0010512, 0.0013111
4: -0.0005992, 0.0013043, -0.0005591, 0.0013039, -0.0010599, 0.0009391
5: 0.0027464, 0.0046501, 0.0028046, 0.0046253, -0.0011644, 0.0010897
6: -0.0110161, -0.0038502, -0.0109548, -0.0039484, -0.0037016, 0.0039606
7: 0.0023202, 0.0122567, 0.0026119, 0.0122551, -0.0056378, 0.0049270
8: 0.9911066, 0.9978906, 0.9912009, 0.9978716, -0.0037318, 0.0033844
9: -0.0139336, -0.0076946, -0.0139326, -0.0078321, -0.0030800, 0.0034818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025249, upper bound: 0.0023835
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025249, upper bound: 0.0023729
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058600, 0.0096025, 0.0059470, 0.0091044, -0.0021116, 0.0028107
1: 0.0015418, 0.0026271, 0.0019807, 0.0026121, -0.0009577, 0.0004715
2: 0.0089054, 0.0111200, 0.0092937, 0.0110719, -0.0018038, 0.0012412
3: -0.0049932, -0.0026020, -0.0049336, -0.0030464, -0.0011638, 0.0016967
4: -0.0006482, 0.0013685, -0.0005591, 0.0013039, -0.0011808, 0.0010610
5: 0.0025996, 0.0046763, 0.0028046, 0.0046253, -0.0014222, 0.0011781
6: -0.0113321, -0.0037462, -0.0109548, -0.0039484, -0.0042387, 0.0043115
7: 0.0019290, 0.0125854, 0.0026119, 0.0122551, -0.0064417, 0.0055503
8: 0.9910069, 0.9981457, 0.9912009, 0.9978716, -0.0040685, 0.0038450
9: -0.0141438, -0.0075243, -0.0139326, -0.0078321, -0.0034786, 0.0038906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025249, upper bound: 0.0023834
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025249, upper bound: 0.0023729
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0060926, 0.0089290, 0.0061934, 0.0092548, -0.0025083, 0.0017853
1: 0.0022025, 0.0026123, 0.0017756, 0.0026132, -0.0002696, 0.0007309
2: 0.0094232, 0.0109914, 0.0091585, 0.0109357, -0.0009870, 0.0015404
3: -0.0049345, -0.0033126, -0.0049382, -0.0029507, -0.0014637, 0.0010669
4: -0.0004509, 0.0013049, -0.0004286, 0.0013089, -0.0011550, 0.0011447
5: 0.0028784, 0.0045400, 0.0027511, 0.0044810, -0.0010458, 0.0013393
6: -0.0108796, -0.0042868, -0.0110142, -0.0045211, -0.0041495, 0.0045364
7: 0.0032815, 0.0122603, 0.0031384, 0.0122806, -0.0059062, 0.0061208
8: 0.9915254, 0.9978504, 0.9917502, 0.9979010, -0.0042153, 0.0039809
9: -0.0139359, -0.0081947, -0.0139489, -0.0082484, -0.0037627, 0.0037766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023849, upper bound: 0.0023190
time: 0.86 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023793, upper bound: 0.0023189
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0060512, 0.0090268, 0.0061934, 0.0092548, -0.0025327, 0.0016785
1: 0.0021965, 0.0026264, 0.0017756, 0.0026132, -0.0002507, 0.0007171
2: 0.0093692, 0.0110143, 0.0091585, 0.0109357, -0.0009280, 0.0015676
3: -0.0049904, -0.0032889, -0.0049382, -0.0029507, -0.0013945, 0.0009922
4: -0.0004765, 0.0013655, -0.0004286, 0.0013089, -0.0010741, 0.0010768
5: 0.0028212, 0.0045643, 0.0027511, 0.0044810, -0.0009833, 0.0013318
6: -0.0111068, -0.0041906, -0.0110142, -0.0045211, -0.0039013, 0.0042857
7: 0.0031505, 0.0125698, 0.0031384, 0.0122806, -0.0054926, 0.0057547
8: 0.9914331, 0.9980683, 0.9917502, 0.9979010, -0.0039394, 0.0037428
9: -0.0141338, -0.0081109, -0.0139489, -0.0082484, -0.0035391, 0.0035121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023849, upper bound: 0.0022986
time: 0.81 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023793, upper bound: 0.0022986
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058390, 0.0095079, 0.0061790, 0.0094991, -0.0032008, 0.0028710
1: 0.0015436, 0.0026122, 0.0015568, 0.0026131, -0.0009691, 0.0009601
2: 0.0089566, 0.0111316, 0.0089506, 0.0109437, -0.0018203, 0.0020247
3: -0.0049342, -0.0026014, -0.0049379, -0.0027326, -0.0016585, 0.0017863
4: -0.0006604, 0.0013046, -0.0004564, 0.0013086, -0.0013283, 0.0011302
5: 0.0026472, 0.0046886, 0.0026705, 0.0044894, -0.0014649, 0.0016491
6: -0.0111093, -0.0036975, -0.0110703, -0.0044877, -0.0043850, 0.0051110
7: 0.0018745, 0.0122587, 0.0028695, 0.0122788, -0.0071498, 0.0061606
8: 0.9909601, 0.9979205, 0.9917182, 0.9979175, -0.0046824, 0.0039673
9: -0.0139349, -0.0074846, -0.0139477, -0.0081484, -0.0037235, 0.0043691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025083, upper bound: 0.0023929
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025083, upper bound: 0.0023838
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058759, 0.0094484, 0.0062225, 0.0094302, -0.0030330, 0.0027754
1: 0.0016071, 0.0026122, 0.0016663, 0.0026224, -0.0009179, 0.0008539
2: 0.0090152, 0.0111113, 0.0090228, 0.0109196, -0.0017470, 0.0018798
3: -0.0049340, -0.0026735, -0.0049747, -0.0028537, -0.0015646, 0.0017684
4: -0.0006324, 0.0013044, -0.0004216, 0.0013484, -0.0013667, 0.0011392
5: 0.0026699, 0.0046670, 0.0026805, 0.0044639, -0.0014321, 0.0015974
6: -0.0110882, -0.0037831, -0.0111770, -0.0045889, -0.0044155, 0.0052077
7: 0.0020472, 0.0122574, 0.0031022, 0.0124826, -0.0073165, 0.0061433
8: 0.9910423, 0.9979132, 0.9918153, 0.9980462, -0.0048162, 0.0040218
9: -0.0139340, -0.0075782, -0.0140781, -0.0082661, -0.0037487, 0.0044933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025083, upper bound: 0.0023970
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025083, upper bound: 0.0023754
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058754, 0.0094021, 0.0061085, 0.0090259, -0.0018792, 0.0024325
1: 0.0016631, 0.0026123, 0.0022048, 0.0026263, -0.0008420, 0.0002057
2: 0.0090547, 0.0111115, 0.0093697, 0.0109826, -0.0015534, 0.0010390
3: -0.0049344, -0.0027205, -0.0049899, -0.0033217, -0.0008143, 0.0016065
4: -0.0006281, 0.0013048, -0.0004410, 0.0013649, -0.0012102, 0.0008816
5: 0.0026918, 0.0046672, 0.0028217, 0.0045307, -0.0012360, 0.0011008
6: -0.0110698, -0.0037822, -0.0111046, -0.0043239, -0.0036261, 0.0043678
7: 0.0020990, 0.0122596, 0.0033321, 0.0125668, -0.0065017, 0.0045080
8: 0.9910414, 0.9979084, 0.9915611, 0.9980661, -0.0041903, 0.0032572
9: -0.0139355, -0.0075949, -0.0141319, -0.0082270, -0.0028826, 0.0039796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024330, upper bound: 0.0022755
time: 0.75 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024330, upper bound: 0.0022703
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0059143, 0.0093406, 0.0061687, 0.0090854, -0.0019546, 0.0023184
1: 0.0017230, 0.0026122, 0.0022135, 0.0026349, -0.0007901, 0.0002080
2: 0.0091041, 0.0110900, 0.0093368, 0.0109494, -0.0014651, 0.0010806
3: -0.0049342, -0.0027940, -0.0050239, -0.0033561, -0.0008232, 0.0015879
4: -0.0005995, 0.0013046, -0.0004038, 0.0014017, -0.0012522, 0.0008911
5: 0.0027142, 0.0046445, 0.0027869, 0.0044955, -0.0011924, 0.0011450
6: -0.0110486, -0.0038725, -0.0112429, -0.0044636, -0.0036212, 0.0045430
7: 0.0022737, 0.0122585, 0.0035224, 0.0127551, -0.0066840, 0.0045569
8: 0.9911279, 0.9979011, 0.9916951, 0.9981989, -0.0043584, 0.0032822
9: -0.0139348, -0.0076906, -0.0142523, -0.0083487, -0.0029138, 0.0041149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024330, upper bound: 0.0022777
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024330, upper bound: 0.0022667
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0057960, 0.0097399, 0.0059598, 0.0093507, -0.0025297, 0.0032184
1: 0.0013617, 0.0026123, 0.0017950, 0.0026269, -0.0011641, 0.0006489
2: 0.0087411, 0.0111554, 0.0091258, 0.0110649, -0.0021053, 0.0015344
3: -0.0049345, -0.0024210, -0.0049924, -0.0028846, -0.0012418, 0.0019736
4: -0.0007008, 0.0013049, -0.0005668, 0.0013676, -0.0013108, 0.0009464
5: 0.0025692, 0.0047138, 0.0026868, 0.0046178, -0.0015790, 0.0013694
6: -0.0111686, -0.0035974, -0.0112480, -0.0039781, -0.0040231, 0.0048060
7: 0.0015766, 0.0122602, 0.0024743, 0.0125807, -0.0071982, 0.0050754
8: 0.9908641, 0.9979401, 0.9912293, 0.9981176, -0.0045022, 0.0034378
9: -0.0139359, -0.0073460, -0.0141408, -0.0077995, -0.0031121, 0.0043218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025341, upper bound: 0.0023647
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025344, upper bound: 0.0023797
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058338, 0.0096626, 0.0060137, 0.0093266, -0.0024301, 0.0031015
1: 0.0014276, 0.0026122, 0.0018958, 0.0026354, -0.0011109, 0.0005482
2: 0.0088126, 0.0111345, 0.0091582, 0.0110350, -0.0020173, 0.0014353
3: -0.0049343, -0.0024944, -0.0050261, -0.0029976, -0.0011536, 0.0019512
4: -0.0006725, 0.0013047, -0.0005256, 0.0014041, -0.0013514, 0.0009517
5: 0.0025958, 0.0046916, 0.0026856, 0.0045862, -0.0015325, 0.0013510
6: -0.0111472, -0.0036853, -0.0113558, -0.0041035, -0.0040190, 0.0049431
7: 0.0017520, 0.0122590, 0.0027330, 0.0127676, -0.0073693, 0.0050437
8: 0.9909484, 0.9979326, 0.9913496, 0.9982405, -0.0046590, 0.0034749
9: -0.0139351, -0.0074406, -0.0142603, -0.0079379, -0.0031247, 0.0044518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025341, upper bound: 0.0023659
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025344, upper bound: 0.0023785
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022863, upper bound: 0.0022333
IS_A1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022810, upper bound: 0.0022333
IS_A1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022863, upper bound: 0.0022333
IS_A1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022810, upper bound: 0.0022333
IS_A1_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023563, upper bound: 0.0023647
IS_A1_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023563, upper bound: 0.0023540
IS_A1_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023563, upper bound: 0.0023647
IS_A1_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023563, upper bound: 0.0023540
IS_A1_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022706, upper bound: 0.0023849
IS_A1_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022705, upper bound: 0.0023793
IS_A1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022706, upper bound: 0.0023849
IS_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022705, upper bound: 0.0023793
IS_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023494, upper bound: 0.0025104
IS_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023494, upper bound: 0.0025083
IS_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023494, upper bound: 0.0025104
IS_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023494, upper bound: 0.0025083
IS_A1_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022797, upper bound: 0.0022043
IS_A1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022797, upper bound: 0.0022108
IS_A1_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022797, upper bound: 0.0022090
IS_A1_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022797, upper bound: 0.0022148
IS_A1_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023560, upper bound: 0.0023408
IS_A1_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023560, upper bound: 0.0023537
IS_A1_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023560, upper bound: 0.0023421
IS_A1_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023560, upper bound: 0.0023560
IS_A1_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022628, upper bound: 0.0023594
IS_A1_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022628, upper bound: 0.0023702
IS_A1_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022628, upper bound: 0.0023553
IS_A1_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0022628, upper bound: 0.0023655
IS_A1_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023377, upper bound: 0.0024979
IS_A1_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023377, upper bound: 0.0025161
IS_A1_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023377, upper bound: 0.0024952
IS_A1_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023377, upper bound: 0.0025093
IS_A2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023861, upper bound: 0.0022702
IS_A2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023795, upper bound: 0.0022702
IS_A2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023861, upper bound: 0.0022702
IS_A2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023795, upper bound: 0.0022702
IS_A2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025177, upper bound: 0.0023491
IS_A2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025156, upper bound: 0.0023492
IS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025177, upper bound: 0.0023491
IS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025156, upper bound: 0.0023491
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0024398, upper bound: 0.0022328
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0024409, upper bound: 0.0022328
IS_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0024409, upper bound: 0.0022349
IS_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0024409, upper bound: 0.0022328
IS_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025249, upper bound: 0.0023835
IS_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025249, upper bound: 0.0023729
IS_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025249, upper bound: 0.0023834
IS_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025249, upper bound: 0.0023729
IS_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023849, upper bound: 0.0023190
IS_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023793, upper bound: 0.0023189
IS_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023849, upper bound: 0.0022986
IS_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0023793, upper bound: 0.0022986
IS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025083, upper bound: 0.0023929
IS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025083, upper bound: 0.0023838
IS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025083, upper bound: 0.0023970
IS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025083, upper bound: 0.0023754
IS_A2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0024330, upper bound: 0.0022755
IS_A2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0024330, upper bound: 0.0022703
IS_A2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0024330, upper bound: 0.0022777
IS_A2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0024330, upper bound: 0.0022667
IS_A2_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025341, upper bound: 0.0023647
IS_A2_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025344, upper bound: 0.0023797
IS_A2_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025341, upper bound: 0.0023659
IS_A2_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 8, lower bound: -0.0025344, upper bound: 0.0023785

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0063003, 0.0088814, 0.0063563, 0.0088348, -0.0013627, 0.0014247
1: 0.0021614, 0.0025989, 0.0022406, 0.0025987, -0.0002682, 0.0001934
2: 0.0094413, 0.0108766, 0.0094753, 0.0108456, -0.0008059, 0.0007534
3: -0.0048817, -0.0033620, -0.0048806, -0.0034634, -0.0007656, 0.0008442
4: -0.0003293, 0.0012477, -0.0002876, 0.0012466, -0.0008499, 0.0008288
5: 0.0029137, 0.0044184, 0.0029336, 0.0043855, -0.0008185, 0.0007983
6: -0.0106858, -0.0047696, -0.0106605, -0.0048998, -0.0031459, 0.0031672
7: 0.0038586, 0.0119677, 0.0041164, 0.0119620, -0.0043868, 0.0042382
8: 0.9919887, 0.9976511, 0.9921136, 0.9976402, -0.0030385, 0.0029965
9: -0.0137488, -0.0085889, -0.0137452, -0.0087285, -0.0027100, 0.0027821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022073, upper bound: 0.0021476
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0021522
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0063554, 0.0088969, 0.0063918, 0.0088344, -0.0013834, 0.0014040
1: 0.0022331, 0.0026070, 0.0022457, 0.0025986, -0.0002064, 0.0002015
2: 0.0094403, 0.0108461, 0.0094755, 0.0108260, -0.0007778, 0.0007649
3: -0.0049135, -0.0034555, -0.0048804, -0.0034837, -0.0007975, 0.0007976
4: -0.0002889, 0.0012822, -0.0002657, 0.0012464, -0.0008570, 0.0008633
5: 0.0028980, 0.0043861, 0.0029338, 0.0043648, -0.0008210, 0.0008104
6: -0.0107963, -0.0048977, -0.0106597, -0.0049823, -0.0032461, 0.0032155
7: 0.0041049, 0.0121439, 0.0042287, 0.0119608, -0.0043868, 0.0044147
8: 0.9921115, 0.9977691, 0.9921927, 0.9976393, -0.0030848, 0.0031113
9: -0.0138615, -0.0087238, -0.0137444, -0.0088003, -0.0028229, 0.0028027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021928, upper bound: 0.0021495
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0022038, upper bound: 0.0021522
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0062526, 0.0091878, 0.0063563, 0.0088348, -0.0015105, 0.0020278
1: 0.0018539, 0.0026131, 0.0022406, 0.0025987, -0.0006165, 0.0002198
2: 0.0092166, 0.0109029, 0.0094753, 0.0108456, -0.0012417, 0.0008351
3: -0.0049377, -0.0030541, -0.0048806, -0.0034634, -0.0008699, 0.0012398
4: -0.0003857, 0.0013084, -0.0002876, 0.0012466, -0.0009688, 0.0009417
5: 0.0027753, 0.0044463, 0.0029336, 0.0043855, -0.0010852, 0.0008848
6: -0.0109923, -0.0046589, -0.0106605, -0.0048998, -0.0036959, 0.0035108
7: 0.0033982, 0.0122780, 0.0041164, 0.0119620, -0.0051809, 0.0048154
8: 0.9918824, 0.9978935, 0.9921136, 0.9976402, -0.0033681, 0.0034365
9: -0.0139473, -0.0083916, -0.0137452, -0.0087285, -0.0030791, 0.0031844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022531, upper bound: 0.0021470
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022550, upper bound: 0.0021518
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0063042, 0.0091713, 0.0063918, 0.0088344, -0.0015239, 0.0019506
1: 0.0019491, 0.0026224, 0.0022457, 0.0025986, -0.0005222, 0.0002308
2: 0.0092506, 0.0108744, 0.0094755, 0.0108260, -0.0011589, 0.0008425
3: -0.0049746, -0.0031643, -0.0048804, -0.0034837, -0.0009135, 0.0011572
4: -0.0003463, 0.0013483, -0.0002657, 0.0012464, -0.0009695, 0.0009889
5: 0.0027689, 0.0044161, 0.0029338, 0.0043648, -0.0010761, 0.0008927
6: -0.0111083, -0.0047787, -0.0106597, -0.0049823, -0.0038518, 0.0035419
7: 0.0036469, 0.0124820, 0.0042287, 0.0119608, -0.0051320, 0.0050572
8: 0.9919973, 0.9980263, 0.9921927, 0.9976393, -0.0033979, 0.0036030
9: -0.0140777, -0.0085241, -0.0137444, -0.0088003, -0.0032337, 0.0031828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022563, upper bound: 0.0021470
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022579, upper bound: 0.0021518
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061830, 0.0091586, 0.0062691, 0.0089809, -0.0018322, 0.0021382
1: 0.0018077, 0.0025990, 0.0020183, 0.0025989, -0.0006643, 0.0004178
2: 0.0092143, 0.0109414, 0.0093655, 0.0108938, -0.0013463, 0.0010843
3: -0.0048821, -0.0029770, -0.0048815, -0.0032137, -0.0009594, 0.0012797
4: -0.0004325, 0.0012482, -0.0003616, 0.0012476, -0.0009893, 0.0008425
5: 0.0028079, 0.0044870, 0.0028731, 0.0044366, -0.0011028, 0.0010150
6: -0.0107835, -0.0044971, -0.0107242, -0.0046971, -0.0033625, 0.0036772
7: 0.0031359, 0.0119700, 0.0036105, 0.0119669, -0.0052953, 0.0044294
8: 0.9917272, 0.9976826, 0.9919190, 0.9976631, -0.0034638, 0.0030430
9: -0.0137503, -0.0082370, -0.0137483, -0.0084771, -0.0027637, 0.0032525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022753, upper bound: 0.0022760
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022771, upper bound: 0.0022874
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062212, 0.0091103, 0.0063193, 0.0089768, -0.0017583, 0.0020503
1: 0.0018616, 0.0025990, 0.0021058, 0.0026069, -0.0006177, 0.0003323
2: 0.0092559, 0.0109203, 0.0093809, 0.0108661, -0.0012767, 0.0010083
3: -0.0048819, -0.0030452, -0.0049133, -0.0033175, -0.0008963, 0.0012562
4: -0.0004045, 0.0012480, -0.0003232, 0.0012819, -0.0010153, 0.0008515
5: 0.0028252, 0.0044647, 0.0028652, 0.0044072, -0.0010718, 0.0009993
6: -0.0107683, -0.0045858, -0.0108313, -0.0048138, -0.0033876, 0.0037592
7: 0.0033067, 0.0119689, 0.0038544, 0.0121427, -0.0053998, 0.0044288
8: 0.9918123, 0.9976777, 0.9920311, 0.9977800, -0.0035640, 0.0030909
9: -0.0137496, -0.0083305, -0.0138607, -0.0086063, -0.0027898, 0.0033352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022753, upper bound: 0.0022673
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022771, upper bound: 0.0022771
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061341, 0.0095665, 0.0062691, 0.0089809, -0.0019829, 0.0027898
1: 0.0014963, 0.0026132, 0.0020183, 0.0025989, -0.0010083, 0.0004463
2: 0.0088972, 0.0109685, 0.0093655, 0.0108938, -0.0018201, 0.0011676
3: -0.0049382, -0.0026561, -0.0048815, -0.0032137, -0.0010721, 0.0016956
4: -0.0004894, 0.0013089, -0.0003616, 0.0012476, -0.0011117, 0.0009644
5: 0.0026465, 0.0045157, 0.0028731, 0.0044366, -0.0013848, 0.0011033
6: -0.0110862, -0.0043833, -0.0107242, -0.0046971, -0.0039099, 0.0040273
7: 0.0026711, 0.0122807, 0.0036105, 0.0119669, -0.0061316, 0.0050530
8: 0.9916182, 0.9979236, 0.9919190, 0.9976631, -0.0037997, 0.0035057
9: -0.0139490, -0.0080379, -0.0137483, -0.0084771, -0.0031624, 0.0036680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023314, upper bound: 0.0022754
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023332, upper bound: 0.0022856
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061686, 0.0094916, 0.0063193, 0.0089768, -0.0019136, 0.0026824
1: 0.0015621, 0.0026132, 0.0021058, 0.0026069, -0.0009523, 0.0003607
2: 0.0089567, 0.0109494, 0.0093809, 0.0108661, -0.0017367, 0.0010942
3: -0.0049379, -0.0027313, -0.0049133, -0.0033175, -0.0010086, 0.0016648
4: -0.0004634, 0.0013087, -0.0003232, 0.0012819, -0.0011404, 0.0009730
5: 0.0026729, 0.0044955, 0.0028652, 0.0044072, -0.0013461, 0.0010903
6: -0.0110695, -0.0044636, -0.0108313, -0.0048138, -0.0039307, 0.0041202
7: 0.0028368, 0.0122793, 0.0038544, 0.0121427, -0.0062351, 0.0050503
8: 0.9916950, 0.9979175, 0.9920311, 0.9977800, -0.0039104, 0.0035519
9: -0.0139481, -0.0081253, -0.0138607, -0.0086063, -0.0031873, 0.0037588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023314, upper bound: 0.0022656
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023332, upper bound: 0.0022746
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062558, 0.0089195, 0.0061508, 0.0089282, -0.0016049, 0.0017897
1: 0.0021012, 0.0025990, 0.0022109, 0.0026122, -0.0003617, 0.0002338
2: 0.0094129, 0.0109012, 0.0094237, 0.0109593, -0.0010264, 0.0008873
3: -0.0048820, -0.0032864, -0.0049340, -0.0033459, -0.0009253, 0.0010320
4: -0.0003618, 0.0012481, -0.0004149, 0.0013044, -0.0010046, 0.0010017
5: 0.0028982, 0.0044444, 0.0028789, 0.0045059, -0.0010158, 0.0009401
6: -0.0107022, -0.0046663, -0.0108776, -0.0044221, -0.0038222, 0.0037301
7: 0.0036605, 0.0119695, 0.0034658, 0.0122577, -0.0052095, 0.0051225
8: 0.9918895, 0.9976574, 0.9916552, 0.9978484, -0.0035785, 0.0036280
9: -0.0137499, -0.0084801, -0.0139342, -0.0083125, -0.0032755, 0.0032904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021917, upper bound: 0.0023029
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021933, upper bound: 0.0023096
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062920, 0.0088876, 0.0062061, 0.0089856, -0.0016546, 0.0017356
1: 0.0021518, 0.0025990, 0.0022189, 0.0026205, -0.0003193, 0.0002364
2: 0.0094366, 0.0108812, 0.0093920, 0.0109287, -0.0009807, 0.0009148
3: -0.0048818, -0.0033489, -0.0049669, -0.0033775, -0.0009357, 0.0010178
4: -0.0003353, 0.0012479, -0.0003806, 0.0013400, -0.0010313, 0.0010130
5: 0.0029113, 0.0044232, 0.0028453, 0.0044735, -0.0009984, 0.0009693
6: -0.0106888, -0.0047503, -0.0110110, -0.0045506, -0.0038419, 0.0038458
7: 0.0038229, 0.0119683, 0.0036408, 0.0124393, -0.0053195, 0.0051799
8: 0.9919701, 0.9976525, 0.9917785, 0.9979764, -0.0036895, 0.0036615
9: -0.0137492, -0.0085689, -0.0140504, -0.0084244, -0.0033122, 0.0033757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021917, upper bound: 0.0022954
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021933, upper bound: 0.0023034
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062076, 0.0092383, 0.0061508, 0.0089282, -0.0017617, 0.0024079
1: 0.0017942, 0.0026132, 0.0022109, 0.0026122, -0.0007096, 0.0002600
2: 0.0091725, 0.0109279, 0.0094237, 0.0109593, -0.0014777, 0.0009740
3: -0.0049381, -0.0029754, -0.0049340, -0.0033459, -0.0010290, 0.0014338
4: -0.0004185, 0.0013088, -0.0004149, 0.0013044, -0.0011285, 0.0011140
5: 0.0027570, 0.0044727, 0.0028789, 0.0045059, -0.0012879, 0.0010320
6: -0.0110090, -0.0045541, -0.0108776, -0.0044221, -0.0043728, 0.0040946
7: 0.0032011, 0.0122800, 0.0034658, 0.0122577, -0.0060288, 0.0056964
8: 0.9917819, 0.9978991, 0.9916552, 0.9978484, -0.0039282, 0.0040649
9: -0.0139485, -0.0082823, -0.0139342, -0.0083125, -0.0036425, 0.0037087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022386, upper bound: 0.0023008
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022399, upper bound: 0.0023077
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062431, 0.0091892, 0.0062061, 0.0089856, -0.0018133, 0.0023246
1: 0.0018513, 0.0026131, 0.0022189, 0.0026205, -0.0006586, 0.0002624
2: 0.0092156, 0.0109082, 0.0093920, 0.0109287, -0.0014070, 0.0010025
3: -0.0049378, -0.0030451, -0.0049669, -0.0033775, -0.0010387, 0.0014081
4: -0.0003918, 0.0013085, -0.0003806, 0.0013400, -0.0011562, 0.0011245
5: 0.0027747, 0.0044519, 0.0028453, 0.0044735, -0.0012585, 0.0010622
6: -0.0109933, -0.0046367, -0.0110110, -0.0045506, -0.0043834, 0.0042145
7: 0.0033644, 0.0122786, 0.0036408, 0.0124393, -0.0061359, 0.0057502
8: 0.9918611, 0.9978941, 0.9917785, 0.9979764, -0.0040432, 0.0040958
9: -0.0139476, -0.0083717, -0.0140504, -0.0084244, -0.0036768, 0.0037969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022386, upper bound: 0.0022951
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022399, upper bound: 0.0023022
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061830, 0.0091586, 0.0060066, 0.0090472, -0.0019110, 0.0024693
1: 0.0018077, 0.0025990, 0.0020529, 0.0026119, -0.0006917, 0.0003826
2: 0.0092143, 0.0109414, 0.0093370, 0.0110390, -0.0015293, 0.0010964
3: -0.0048821, -0.0029770, -0.0049331, -0.0031358, -0.0010796, 0.0013881
4: -0.0004325, 0.0012482, -0.0005170, 0.0013035, -0.0011067, 0.0010414
5: 0.0028079, 0.0044870, 0.0028288, 0.0045904, -0.0012968, 0.0010861
6: -0.0107835, -0.0044971, -0.0109272, -0.0040870, -0.0041321, 0.0040944
7: 0.0031359, 0.0119700, 0.0028615, 0.0122527, -0.0058952, 0.0054055
8: 0.9917272, 0.9976826, 0.9913338, 0.9978622, -0.0038800, 0.0037812
9: -0.0137503, -0.0082370, -0.0139310, -0.0079723, -0.0034111, 0.0036362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022673, upper bound: 0.0024272
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022698, upper bound: 0.0024379
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062212, 0.0091103, 0.0060503, 0.0090503, -0.0018581, 0.0023796
1: 0.0018616, 0.0025990, 0.0021305, 0.0026212, -0.0006434, 0.0003112
2: 0.0092559, 0.0109203, 0.0093460, 0.0110148, -0.0014587, 0.0010443
3: -0.0048819, -0.0030452, -0.0049697, -0.0032243, -0.0010291, 0.0013578
4: -0.0004045, 0.0012480, -0.0004836, 0.0013431, -0.0011252, 0.0010503
5: 0.0028252, 0.0044647, 0.0028169, 0.0045648, -0.0012647, 0.0010735
6: -0.0107683, -0.0045858, -0.0110512, -0.0041885, -0.0041529, 0.0041482
7: 0.0033067, 0.0119689, 0.0030730, 0.0124553, -0.0059619, 0.0054120
8: 0.9918123, 0.9976777, 0.9914311, 0.9979970, -0.0039525, 0.0038250
9: -0.0137496, -0.0083305, -0.0140606, -0.0080847, -0.0034373, 0.0036947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022673, upper bound: 0.0024249
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022698, upper bound: 0.0024350
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061341, 0.0095665, 0.0060066, 0.0090472, -0.0020616, 0.0031209
1: 0.0014963, 0.0026132, 0.0020529, 0.0026119, -0.0010357, 0.0004111
2: 0.0088972, 0.0109685, 0.0093370, 0.0110390, -0.0020031, 0.0011797
3: -0.0049382, -0.0026561, -0.0049331, -0.0031358, -0.0011922, 0.0018040
4: -0.0004894, 0.0013089, -0.0005170, 0.0013035, -0.0012291, 0.0011633
5: 0.0026465, 0.0045157, 0.0028288, 0.0045904, -0.0015788, 0.0011744
6: -0.0110862, -0.0043833, -0.0109272, -0.0040870, -0.0046795, 0.0044445
7: 0.0026711, 0.0122807, 0.0028615, 0.0122527, -0.0067315, 0.0060291
8: 0.9916182, 0.9979236, 0.9913338, 0.9978622, -0.0042159, 0.0042440
9: -0.0139490, -0.0080379, -0.0139310, -0.0079723, -0.0038098, 0.0040516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023139, upper bound: 0.0024223
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023155, upper bound: 0.0024297
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061686, 0.0094916, 0.0060503, 0.0090503, -0.0020135, 0.0030116
1: 0.0015621, 0.0026132, 0.0021305, 0.0026212, -0.0009780, 0.0003396
2: 0.0089567, 0.0109494, 0.0093460, 0.0110148, -0.0019187, 0.0011301
3: -0.0049379, -0.0027313, -0.0049697, -0.0032243, -0.0011414, 0.0017663
4: -0.0004634, 0.0013087, -0.0004836, 0.0013431, -0.0012504, 0.0011718
5: 0.0026729, 0.0044955, 0.0028169, 0.0045648, -0.0015389, 0.0011645
6: -0.0110695, -0.0044636, -0.0110512, -0.0041885, -0.0046959, 0.0045092
7: 0.0028368, 0.0122793, 0.0030730, 0.0124553, -0.0067972, 0.0060336
8: 0.9916950, 0.9979175, 0.9914311, 0.9979970, -0.0042988, 0.0042860
9: -0.0139481, -0.0081253, -0.0140606, -0.0080847, -0.0038347, 0.0041182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023139, upper bound: 0.0024189
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023155, upper bound: 0.0024269
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061657, 0.0092586, 0.0064317, 0.0089111, -0.0017291, 0.0022316
1: 0.0016991, 0.0025980, 0.0022515, 0.0026097, -0.0008071, 0.0001803
2: 0.0091313, 0.0109510, 0.0094332, 0.0108039, -0.0014357, 0.0009560
3: -0.0048779, -0.0028654, -0.0049242, -0.0035065, -0.0007137, 0.0015071
4: -0.0004517, 0.0012437, -0.0002409, 0.0012938, -0.0011158, 0.0007726
5: 0.0027741, 0.0044972, 0.0028890, 0.0043414, -0.0011229, 0.0010129
6: -0.0107962, -0.0044567, -0.0108378, -0.0050751, -0.0032325, 0.0040189
7: 0.0029825, 0.0119470, 0.0043551, 0.0122034, -0.0060124, 0.0039509
8: 0.9916885, 0.9976744, 0.9922817, 0.9978102, -0.0038556, 0.0028778
9: -0.0137356, -0.0081699, -0.0138996, -0.0088811, -0.0025263, 0.0036710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021972, upper bound: 0.0021837
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021972, upper bound: 0.0021745
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0061628, 0.0092692, 0.0064198, 0.0089349, -0.0017742, 0.0022644
1: 0.0016914, 0.0025984, 0.0022498, 0.0026131, -0.0008217, 0.0001826
2: 0.0091236, 0.0109526, 0.0094200, 0.0108105, -0.0014566, 0.0009809
3: -0.0048794, -0.0028578, -0.0049378, -0.0034997, -0.0007228, 0.0015411
4: -0.0004537, 0.0012453, -0.0002483, 0.0013086, -0.0011435, 0.0007825
5: 0.0027697, 0.0044989, 0.0028750, 0.0043483, -0.0011388, 0.0010393
6: -0.0108045, -0.0044501, -0.0108931, -0.0050475, -0.0032750, 0.0041238
7: 0.0029706, 0.0119553, 0.0043175, 0.0122787, -0.0061570, 0.0040015
8: 0.9916821, 0.9976811, 0.9922552, 0.9978633, -0.0039562, 0.0029149
9: -0.0137409, -0.0081635, -0.0139477, -0.0088571, -0.0025587, 0.0037608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021989, upper bound: 0.0021933
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021989, upper bound: 0.0021804
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0062007, 0.0091978, 0.0064832, 0.0089746, -0.0018052, 0.0021223
1: 0.0017594, 0.0025979, 0.0022589, 0.0026189, -0.0007555, 0.0001830
2: 0.0091811, 0.0109316, 0.0093980, 0.0107755, -0.0013510, 0.0009980
3: -0.0048777, -0.0029376, -0.0049606, -0.0035360, -0.0007243, 0.0014872
4: -0.0004256, 0.0012435, -0.0002091, 0.0013332, -0.0011581, 0.0007841
5: 0.0027959, 0.0044767, 0.0028517, 0.0043112, -0.0010814, 0.0010575
6: -0.0107786, -0.0045382, -0.0109855, -0.0051948, -0.0032349, 0.0041957
7: 0.0031448, 0.0119460, 0.0045182, 0.0124046, -0.0061891, 0.0040099
8: 0.9917666, 0.9976689, 0.9923966, 0.9979520, -0.0040252, 0.0029085
9: -0.0137350, -0.0082575, -0.0140282, -0.0089854, -0.0025640, 0.0038064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021972, upper bound: 0.0021887
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021972, upper bound: 0.0021731
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0061985, 0.0092088, 0.0064724, 0.0089978, -0.0018384, 0.0021549
1: 0.0017511, 0.0025983, 0.0022574, 0.0026222, -0.0007694, 0.0001852
2: 0.0091729, 0.0109329, 0.0093852, 0.0107814, -0.0013723, 0.0010164
3: -0.0048792, -0.0029295, -0.0049738, -0.0035298, -0.0007330, 0.0015171
4: -0.0004272, 0.0012451, -0.0002157, 0.0013475, -0.0011787, 0.0007935
5: 0.0027914, 0.0044780, 0.0028382, 0.0043175, -0.0010973, 0.0010769
6: -0.0107870, -0.0045329, -0.0110394, -0.0051697, -0.0032751, 0.0042729
7: 0.0031341, 0.0119543, 0.0044840, 0.0124779, -0.0063023, 0.0040577
8: 0.9917616, 0.9976755, 0.9923725, 0.9980036, -0.0040992, 0.0029436
9: -0.0137402, -0.0082522, -0.0140751, -0.0089635, -0.0025946, 0.0038740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021989, upper bound: 0.0021964
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021989, upper bound: 0.0021757
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0060931, 0.0096214, 0.0063052, 0.0090972, -0.0021636, 0.0030127
1: 0.0013875, 0.0025980, 0.0019502, 0.0026103, -0.0011363, 0.0004879
2: 0.0088177, 0.0109912, 0.0092905, 0.0108739, -0.0019918, 0.0012763
3: -0.0048780, -0.0025490, -0.0049267, -0.0031659, -0.0010033, 0.0018821
4: -0.0005219, 0.0012437, -0.0003455, 0.0012964, -0.0012162, 0.0008187
5: 0.0026448, 0.0045397, 0.0028135, 0.0044155, -0.0014612, 0.0012016
6: -0.0108834, -0.0042880, -0.0109227, -0.0047810, -0.0035837, 0.0044001
7: 0.0024607, 0.0119473, 0.0036511, 0.0122169, -0.0067110, 0.0043483
8: 0.9915267, 0.9976985, 0.9919996, 0.9978430, -0.0041490, 0.0030176
9: -0.0137358, -0.0079285, -0.0139081, -0.0085268, -0.0026888, 0.0040122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022721, upper bound: 0.0023139
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022721, upper bound: 0.0023088
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060893, 0.0096428, 0.0062834, 0.0091638, -0.0022881, 0.0030684
1: 0.0013726, 0.0025984, 0.0018852, 0.0026138, -0.0011572, 0.0005597
2: 0.0087995, 0.0109932, 0.0092392, 0.0108860, -0.0020277, 0.0013706
3: -0.0048795, -0.0025326, -0.0049405, -0.0030935, -0.0010830, 0.0019218
4: -0.0005251, 0.0012454, -0.0003640, 0.0013114, -0.0012404, 0.0008377
5: 0.0026364, 0.0045419, 0.0027831, 0.0044283, -0.0014887, 0.0012547
6: -0.0108941, -0.0042792, -0.0109965, -0.0047302, -0.0036471, 0.0045195
7: 0.0024381, 0.0119557, 0.0035260, 0.0122932, -0.0068385, 0.0044833
8: 0.9915181, 0.9977059, 0.9919509, 0.9979016, -0.0042463, 0.0030700
9: -0.0137411, -0.0079173, -0.0139569, -0.0084639, -0.0027539, 0.0040914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022746, upper bound: 0.0023316
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022746, upper bound: 0.0023150
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0061278, 0.0095435, 0.0063490, 0.0090867, -0.0021041, 0.0028988
1: 0.0014536, 0.0025980, 0.0020497, 0.0026195, -0.0010826, 0.0003880
2: 0.0088864, 0.0109719, 0.0093137, 0.0108497, -0.0019042, 0.0012045
3: -0.0048778, -0.0026232, -0.0049631, -0.0032785, -0.0009217, 0.0018578
4: -0.0004954, 0.0012436, -0.0003102, 0.0013359, -0.0012570, 0.0008253
5: 0.0026734, 0.0045194, 0.0028057, 0.0043898, -0.0014168, 0.0012002
6: -0.0108650, -0.0043688, -0.0110397, -0.0048828, -0.0035856, 0.0045406
7: 0.0026238, 0.0119463, 0.0038859, 0.0124185, -0.0068764, 0.0043272
8: 0.9916041, 0.9976931, 0.9920972, 0.9979753, -0.0043075, 0.0030596
9: -0.0137352, -0.0080172, -0.0140371, -0.0086461, -0.0027067, 0.0041422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022721, upper bound: 0.0023165
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022721, upper bound: 0.0022978
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0061240, 0.0095632, 0.0063314, 0.0091423, -0.0021977, 0.0029524
1: 0.0014396, 0.0025983, 0.0019931, 0.0026229, -0.0011022, 0.0004521
2: 0.0088700, 0.0109740, 0.0092747, 0.0108594, -0.0019395, 0.0012744
3: -0.0048793, -0.0026081, -0.0049766, -0.0032177, -0.0009904, 0.0018956
4: -0.0004991, 0.0012452, -0.0003260, 0.0013505, -0.0012780, 0.0008440
5: 0.0026656, 0.0045216, 0.0027773, 0.0044001, -0.0014437, 0.0012386
6: -0.0108753, -0.0043599, -0.0111076, -0.0048419, -0.0036493, 0.0046334
7: 0.0026012, 0.0119547, 0.0037741, 0.0124933, -0.0069957, 0.0044538
8: 0.9915956, 0.9977003, 0.9920580, 0.9980317, -0.0043857, 0.0031132
9: -0.0137405, -0.0080049, -0.0140849, -0.0085922, -0.0027701, 0.0042125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022746, upper bound: 0.0023332
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022746, upper bound: 0.0023041
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061657, 0.0092586, 0.0061844, 0.0090082, -0.0018847, 0.0025505
1: 0.0016991, 0.0025980, 0.0022158, 0.0026237, -0.0008296, 0.0002274
2: 0.0091313, 0.0109510, 0.0093795, 0.0109407, -0.0016132, 0.0010420
3: -0.0048779, -0.0028654, -0.0049798, -0.0033651, -0.0009002, 0.0015961
4: -0.0004517, 0.0012437, -0.0003940, 0.0013539, -0.0012121, 0.0009745
5: 0.0027741, 0.0044972, 0.0028321, 0.0044862, -0.0013098, 0.0011041
6: -0.0107962, -0.0044567, -0.0110635, -0.0045003, -0.0039705, 0.0043806
7: 0.0029825, 0.0119470, 0.0035723, 0.0125108, -0.0065050, 0.0049834
8: 0.9916885, 0.9976744, 0.9917302, 0.9980267, -0.0042025, 0.0035893
9: -0.0137356, -0.0081699, -0.0140961, -0.0083806, -0.0031865, 0.0039859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021906, upper bound: 0.0023359
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021906, upper bound: 0.0023299
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0061628, 0.0092692, 0.0061699, 0.0090293, -0.0019236, 0.0025833
1: 0.0016914, 0.0025984, 0.0022137, 0.0026268, -0.0008433, 0.0002298
2: 0.0091236, 0.0109526, 0.0093678, 0.0109487, -0.0016340, 0.0010635
3: -0.0048794, -0.0028578, -0.0049918, -0.0033568, -0.0009097, 0.0016265
4: -0.0004537, 0.0012453, -0.0004030, 0.0013670, -0.0012359, 0.0009848
5: 0.0027697, 0.0044989, 0.0028197, 0.0044947, -0.0013258, 0.0011268
6: -0.0108045, -0.0044501, -0.0111125, -0.0044666, -0.0040132, 0.0044709
7: 0.0029706, 0.0119553, 0.0035264, 0.0125776, -0.0066298, 0.0050361
8: 0.9916821, 0.9976811, 0.9916980, 0.9980738, -0.0042892, 0.0036275
9: -0.0137409, -0.0081635, -0.0141388, -0.0083512, -0.0032202, 0.0040631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021925, upper bound: 0.0023489
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021925, upper bound: 0.0023327
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0062007, 0.0091978, 0.0062431, 0.0090668, -0.0019529, 0.0024382
1: 0.0017594, 0.0025979, 0.0022242, 0.0026322, -0.0007769, 0.0002297
2: 0.0091811, 0.0109316, 0.0093470, 0.0109082, -0.0015268, 0.0010797
3: -0.0048777, -0.0029376, -0.0050133, -0.0033987, -0.0009093, 0.0015717
4: -0.0004256, 0.0012435, -0.0003577, 0.0013902, -0.0012496, 0.0009844
5: 0.0027959, 0.0044767, 0.0027977, 0.0044518, -0.0012671, 0.0011440
6: -0.0107786, -0.0045382, -0.0111998, -0.0046367, -0.0039671, 0.0045390
7: 0.0031448, 0.0119460, 0.0037581, 0.0126965, -0.0066565, 0.0050339
8: 0.9917666, 0.9976689, 0.9918611, 0.9981575, -0.0043545, 0.0036156
9: -0.0137350, -0.0082575, -0.0142148, -0.0084994, -0.0032188, 0.0041054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021906, upper bound: 0.0023325
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021906, upper bound: 0.0023231
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0061985, 0.0092088, 0.0062316, 0.0090880, -0.0019744, 0.0024724
1: 0.0017511, 0.0025983, 0.0022226, 0.0026353, -0.0007891, 0.0002323
2: 0.0091729, 0.0109329, 0.0093353, 0.0109146, -0.0015488, 0.0010916
3: -0.0048792, -0.0029295, -0.0050254, -0.0033921, -0.0009193, 0.0015949
4: -0.0004272, 0.0012451, -0.0003648, 0.0014033, -0.0012629, 0.0009951
5: 0.0027914, 0.0044780, 0.0027853, 0.0044586, -0.0012841, 0.0011566
6: -0.0107870, -0.0045329, -0.0112490, -0.0046099, -0.0040119, 0.0045890
7: 0.0031341, 0.0119543, 0.0037215, 0.0127635, -0.0067329, 0.0050889
8: 0.9917616, 0.9976755, 0.9918354, 0.9982048, -0.0044025, 0.0036554
9: -0.0137402, -0.0082522, -0.0142577, -0.0084760, -0.0032540, 0.0041494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021925, upper bound: 0.0023455
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021925, upper bound: 0.0023267
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0060931, 0.0096214, 0.0060425, 0.0091640, -0.0022108, 0.0033471
1: 0.0013875, 0.0025980, 0.0020043, 0.0026243, -0.0011579, 0.0004326
2: 0.0088177, 0.0109912, 0.0092645, 0.0110191, -0.0021767, 0.0012675
3: -0.0048780, -0.0025490, -0.0049820, -0.0031107, -0.0011001, 0.0019678
4: -0.0005219, 0.0012437, -0.0004990, 0.0013563, -0.0013089, 0.0010180
5: 0.0026448, 0.0045397, 0.0027669, 0.0045694, -0.0016571, 0.0012576
6: -0.0108834, -0.0042880, -0.0111391, -0.0041705, -0.0043609, 0.0047332
7: 0.0024607, 0.0119473, 0.0029267, 0.0125231, -0.0071851, 0.0053133
8: 0.9915267, 0.9976985, 0.9914139, 0.9980567, -0.0044796, 0.0037633
9: -0.0137358, -0.0079285, -0.0141039, -0.0080291, -0.0033366, 0.0043154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022667, upper bound: 0.0024747
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022667, upper bound: 0.0024512
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060893, 0.0096428, 0.0060258, 0.0092220, -0.0023052, 0.0034003
1: 0.0013726, 0.0025984, 0.0019526, 0.0026279, -0.0011788, 0.0004899
2: 0.0087995, 0.0109932, 0.0092252, 0.0110284, -0.0022112, 0.0013360
3: -0.0048795, -0.0025326, -0.0049965, -0.0030558, -0.0011577, 0.0020073
4: -0.0005251, 0.0012454, -0.0005138, 0.0013720, -0.0013330, 0.0010344
5: 0.0026364, 0.0045419, 0.0027377, 0.0045792, -0.0016832, 0.0013003
6: -0.0108941, -0.0042792, -0.0112100, -0.0041315, -0.0044184, 0.0048384
7: 0.0024381, 0.0119557, 0.0028232, 0.0126033, -0.0073117, 0.0054254
8: 0.9915181, 0.9977059, 0.9913765, 0.9981170, -0.0045728, 0.0038100
9: -0.0137411, -0.0079173, -0.0141553, -0.0079787, -0.0033923, 0.0043939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022692, upper bound: 0.0024946
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022692, upper bound: 0.0024595
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0061278, 0.0095435, 0.0060936, 0.0091527, -0.0021706, 0.0032300
1: 0.0014536, 0.0025980, 0.0020994, 0.0026328, -0.0011025, 0.0003408
2: 0.0088864, 0.0109719, 0.0092853, 0.0109909, -0.0020874, 0.0012198
3: -0.0048778, -0.0026232, -0.0050157, -0.0032168, -0.0010321, 0.0019363
4: -0.0004954, 0.0012436, -0.0004600, 0.0013928, -0.0013420, 0.0010234
5: 0.0026734, 0.0045194, 0.0027607, 0.0045394, -0.0016109, 0.0012530
6: -0.0108650, -0.0043688, -0.0112464, -0.0042892, -0.0043555, 0.0048373
7: 0.0026238, 0.0119463, 0.0031732, 0.0127096, -0.0073111, 0.0052931
8: 0.9916041, 0.9976931, 0.9915277, 0.9981788, -0.0046080, 0.0037983
9: -0.0137352, -0.0080172, -0.0142232, -0.0081602, -0.0033508, 0.0044201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022667, upper bound: 0.0024723
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022667, upper bound: 0.0024504
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0061240, 0.0095632, 0.0060818, 0.0092029, -0.0022384, 0.0032815
1: 0.0014396, 0.0025983, 0.0020552, 0.0026363, -0.0011205, 0.0003893
2: 0.0088700, 0.0109740, 0.0092520, 0.0109974, -0.0021214, 0.0012679
3: -0.0048793, -0.0026081, -0.0050297, -0.0031714, -0.0010831, 0.0019678
4: -0.0004991, 0.0012452, -0.0004709, 0.0014080, -0.0013563, 0.0010401
5: 0.0026656, 0.0045216, 0.0027359, 0.0045464, -0.0016365, 0.0012848
6: -0.0108753, -0.0043599, -0.0113132, -0.0042617, -0.0044142, 0.0049067
7: 0.0026012, 0.0119547, 0.0030932, 0.0127872, -0.0073957, 0.0054035
8: 0.9915956, 0.9977003, 0.9915015, 0.9982367, -0.0046630, 0.0038469
9: -0.0137405, -0.0080049, -0.0142728, -0.0081227, -0.0034073, 0.0044683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022692, upper bound: 0.0024865
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022692, upper bound: 0.0024565
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0061508, 0.0089282, 0.0062558, 0.0089195, -0.0017897, 0.0016049
1: 0.0022109, 0.0026122, 0.0021012, 0.0025990, -0.0002338, 0.0003617
2: 0.0094237, 0.0109593, 0.0094129, 0.0109012, -0.0008873, 0.0010264
3: -0.0049340, -0.0033459, -0.0048820, -0.0032864, -0.0010320, 0.0009253
4: -0.0004149, 0.0013044, -0.0003618, 0.0012481, -0.0010017, 0.0010046
5: 0.0028789, 0.0045059, 0.0028982, 0.0044444, -0.0009401, 0.0010158
6: -0.0108776, -0.0044221, -0.0107022, -0.0046663, -0.0037301, 0.0038222
7: 0.0034658, 0.0122577, 0.0036605, 0.0119695, -0.0051225, 0.0052095
8: 0.9916552, 0.9978484, 0.9918895, 0.9976574, -0.0036280, 0.0035785
9: -0.0139342, -0.0083125, -0.0137499, -0.0084801, -0.0032904, 0.0032755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023029, upper bound: 0.0021917
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023096, upper bound: 0.0021933
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0062061, 0.0089856, 0.0062920, 0.0088876, -0.0017356, 0.0016546
1: 0.0022189, 0.0026205, 0.0021518, 0.0025990, -0.0002364, 0.0003193
2: 0.0093920, 0.0109287, 0.0094366, 0.0108812, -0.0009148, 0.0009807
3: -0.0049669, -0.0033775, -0.0048818, -0.0033489, -0.0010178, 0.0009357
4: -0.0003806, 0.0013400, -0.0003353, 0.0012479, -0.0010130, 0.0010313
5: 0.0028453, 0.0044735, 0.0029113, 0.0044232, -0.0009693, 0.0009984
6: -0.0110110, -0.0045506, -0.0106888, -0.0047503, -0.0038458, 0.0038419
7: 0.0036408, 0.0124393, 0.0038229, 0.0119683, -0.0051799, 0.0053195
8: 0.9917785, 0.9979764, 0.9919701, 0.9976525, -0.0036615, 0.0036895
9: -0.0140504, -0.0084244, -0.0137492, -0.0085689, -0.0033757, 0.0033122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022953, upper bound: 0.0021917
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023034, upper bound: 0.0021933
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0061085, 0.0090259, 0.0062558, 0.0089195, -0.0019040, 0.0017548
1: 0.0022048, 0.0026263, 0.0021012, 0.0025990, -0.0002503, 0.0003834
2: 0.0093697, 0.0109826, 0.0094129, 0.0109012, -0.0009702, 0.0010896
3: -0.0049899, -0.0033217, -0.0048820, -0.0032864, -0.0011178, 0.0009907
4: -0.0004410, 0.0013649, -0.0003618, 0.0012481, -0.0010725, 0.0010974
5: 0.0028217, 0.0045307, 0.0028982, 0.0044444, -0.0010279, 0.0010828
6: -0.0111046, -0.0043239, -0.0107022, -0.0046663, -0.0040786, 0.0040878
7: 0.0033321, 0.0125668, 0.0036605, 0.0119695, -0.0054843, 0.0056841
8: 0.9915611, 0.9980661, 0.9918895, 0.9976574, -0.0038829, 0.0039128
9: -0.0141319, -0.0082270, -0.0137499, -0.0084801, -0.0035938, 0.0035068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023358, upper bound: 0.0021906
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023488, upper bound: 0.0021925
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0061687, 0.0090854, 0.0062920, 0.0088876, -0.0018477, 0.0018183
1: 0.0022135, 0.0026349, 0.0021518, 0.0025990, -0.0002526, 0.0003429
2: 0.0093368, 0.0109494, 0.0094366, 0.0108812, -0.0010053, 0.0010426
3: -0.0050239, -0.0033561, -0.0048818, -0.0033489, -0.0011114, 0.0009998
4: -0.0004038, 0.0014017, -0.0003353, 0.0012479, -0.0010823, 0.0011326
5: 0.0027869, 0.0044955, 0.0029113, 0.0044232, -0.0010651, 0.0010640
6: -0.0112429, -0.0044636, -0.0106888, -0.0047503, -0.0042262, 0.0041024
7: 0.0035224, 0.0127551, 0.0038229, 0.0119683, -0.0055347, 0.0058376
8: 0.9916951, 0.9981989, 0.9919701, 0.9976525, -0.0039114, 0.0040544
9: -0.0142523, -0.0083487, -0.0137492, -0.0085689, -0.0037070, 0.0035390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023325, upper bound: 0.0021906
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023455, upper bound: 0.0021925
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0060066, 0.0090472, 0.0061830, 0.0091586, -0.0024693, 0.0019110
1: 0.0020529, 0.0026119, 0.0018077, 0.0025990, -0.0003826, 0.0006917
2: 0.0093370, 0.0110390, 0.0092143, 0.0109414, -0.0010964, 0.0015293
3: -0.0049331, -0.0031358, -0.0048821, -0.0029770, -0.0013881, 0.0010796
4: -0.0005170, 0.0013035, -0.0004325, 0.0012482, -0.0010414, 0.0011067
5: 0.0028288, 0.0045904, 0.0028079, 0.0044870, -0.0010861, 0.0012968
6: -0.0109272, -0.0040870, -0.0107835, -0.0044971, -0.0040944, 0.0041321
7: 0.0028615, 0.0122527, 0.0031359, 0.0119700, -0.0054055, 0.0058952
8: 0.9913338, 0.9978622, 0.9917272, 0.9976826, -0.0037812, 0.0038800
9: -0.0139310, -0.0079723, -0.0137503, -0.0082370, -0.0036362, 0.0034111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024272, upper bound: 0.0022673
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024379, upper bound: 0.0022699
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0060503, 0.0090503, 0.0062212, 0.0091103, -0.0023796, 0.0018581
1: 0.0021305, 0.0026212, 0.0018616, 0.0025990, -0.0003112, 0.0006434
2: 0.0093460, 0.0110148, 0.0092559, 0.0109203, -0.0010443, 0.0014587
3: -0.0049697, -0.0032243, -0.0048819, -0.0030452, -0.0013578, 0.0010291
4: -0.0004836, 0.0013431, -0.0004045, 0.0012480, -0.0010503, 0.0011252
5: 0.0028169, 0.0045648, 0.0028252, 0.0044647, -0.0010735, 0.0012647
6: -0.0110512, -0.0041885, -0.0107683, -0.0045858, -0.0041482, 0.0041529
7: 0.0030730, 0.0124553, 0.0033067, 0.0119689, -0.0054120, 0.0059619
8: 0.9914311, 0.9979970, 0.9918123, 0.9976777, -0.0038250, 0.0039525
9: -0.0140606, -0.0080847, -0.0137496, -0.0083305, -0.0036947, 0.0034373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024249, upper bound: 0.0022673
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024349, upper bound: 0.0022699
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0059598, 0.0093507, 0.0061830, 0.0091586, -0.0025812, 0.0023974
1: 0.0017950, 0.0026269, 0.0018077, 0.0025990, -0.0006752, 0.0007123
2: 0.0091258, 0.0110649, 0.0092143, 0.0109414, -0.0014511, 0.0015912
3: -0.0049924, -0.0028846, -0.0048821, -0.0029770, -0.0014693, 0.0014013
4: -0.0005668, 0.0013676, -0.0004325, 0.0012482, -0.0011352, 0.0011947
5: 0.0026868, 0.0046178, 0.0028079, 0.0044870, -0.0012971, 0.0013623
6: -0.0112480, -0.0039781, -0.0107835, -0.0044971, -0.0045360, 0.0043922
7: 0.0024743, 0.0125807, 0.0031359, 0.0119700, -0.0060423, 0.0063451
8: 0.9912293, 0.9981176, 0.9917272, 0.9976826, -0.0040308, 0.0042300
9: -0.0141408, -0.0077995, -0.0137503, -0.0082370, -0.0039239, 0.0037297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024747, upper bound: 0.0022666
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024946, upper bound: 0.0022693
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0060137, 0.0093266, 0.0062212, 0.0091103, -0.0024883, 0.0022927
1: 0.0018958, 0.0026354, 0.0018616, 0.0025990, -0.0005771, 0.0006664
2: 0.0091582, 0.0110350, 0.0092559, 0.0109203, -0.0013552, 0.0015188
3: -0.0050261, -0.0029976, -0.0048819, -0.0030452, -0.0014488, 0.0013193
4: -0.0005256, 0.0014041, -0.0004045, 0.0012480, -0.0011398, 0.0012238
5: 0.0026856, 0.0045862, 0.0028252, 0.0044647, -0.0012728, 0.0013283
6: -0.0113558, -0.0041035, -0.0107683, -0.0045858, -0.0046178, 0.0044055
7: 0.0027330, 0.0127676, 0.0033067, 0.0119689, -0.0060112, 0.0064659
8: 0.9913496, 0.9982405, 0.9918123, 0.9976777, -0.0040674, 0.0043403
9: -0.0142603, -0.0079379, -0.0137496, -0.0083305, -0.0040170, 0.0037406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024723, upper bound: 0.0022666
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024866, upper bound: 0.0022693
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0060409, 0.0089537, 0.0061065, 0.0089289, -0.0014585, 0.0014611
1: 0.0021648, 0.0026120, 0.0022045, 0.0026123, -0.0002430, 0.0002060
2: 0.0094049, 0.0110200, 0.0094233, 0.0109837, -0.0008149, 0.0008064
3: -0.0049333, -0.0032528, -0.0049344, -0.0033206, -0.0008153, 0.0008651
4: -0.0004859, 0.0013036, -0.0004422, 0.0013048, -0.0009059, 0.0008826
5: 0.0028684, 0.0045703, 0.0028785, 0.0045319, -0.0008502, 0.0008544
6: -0.0108872, -0.0041667, -0.0108791, -0.0043192, -0.0033308, 0.0033899
7: 0.0030827, 0.0122536, 0.0033257, 0.0122597, -0.0046525, 0.0045135
8: 0.9914103, 0.9978497, 0.9915566, 0.9978499, -0.0032521, 0.0031849
9: -0.0139316, -0.0080786, -0.0139355, -0.0082229, -0.0028861, 0.0029637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023606, upper bound: 0.0021626
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023741, upper bound: 0.0021636
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0060872, 0.0089911, 0.0061433, 0.0089285, -0.0014819, 0.0014795
1: 0.0022017, 0.0026213, 0.0022098, 0.0026122, -0.0002141, 0.0002137
2: 0.0093889, 0.0109944, 0.0094235, 0.0109634, -0.0008180, 0.0008193
3: -0.0049700, -0.0033095, -0.0049342, -0.0033416, -0.0008460, 0.0008474
4: -0.0004542, 0.0013434, -0.0004195, 0.0013046, -0.0009173, 0.0009158
5: 0.0028421, 0.0045432, 0.0028788, 0.0045103, -0.0008667, 0.0008681
6: -0.0110238, -0.0042742, -0.0108782, -0.0044048, -0.0034387, 0.0034443
7: 0.0032644, 0.0124567, 0.0034422, 0.0122585, -0.0046909, 0.0046833
8: 0.9915134, 0.9979886, 0.9916387, 0.9978490, -0.0033043, 0.0032990
9: -0.0140615, -0.0081837, -0.0139347, -0.0082974, -0.0029946, 0.0029995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023628, upper bound: 0.0021626
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023727, upper bound: 0.0021636
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0059536, 0.0093088, 0.0061508, 0.0089282, -0.0016542, 0.0020482
1: 0.0018427, 0.0026271, 0.0022109, 0.0026122, -0.0006172, 0.0002262
2: 0.0091595, 0.0110683, 0.0094237, 0.0109593, -0.0012470, 0.0009145
3: -0.0049930, -0.0029249, -0.0049340, -0.0033459, -0.0008953, 0.0012999
4: -0.0005660, 0.0013683, -0.0004149, 0.0013044, -0.0010561, 0.0009693
5: 0.0027036, 0.0046214, 0.0028789, 0.0045059, -0.0011032, 0.0009690
6: -0.0112341, -0.0039638, -0.0108776, -0.0044221, -0.0037957, 0.0038448
7: 0.0025050, 0.0125841, 0.0034658, 0.0122577, -0.0056156, 0.0049565
8: 0.9912156, 0.9981148, 0.9916552, 0.9978484, -0.0036885, 0.0035358
9: -0.0141430, -0.0078043, -0.0139342, -0.0083125, -0.0031693, 0.0034691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024242, upper bound: 0.0021605
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024249, upper bound: 0.0021670
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0059935, 0.0092612, 0.0062061, 0.0089856, -0.0017282, 0.0019597
1: 0.0019031, 0.0026270, 0.0022189, 0.0026205, -0.0005660, 0.0002280
2: 0.0091964, 0.0110462, 0.0093920, 0.0109287, -0.0011730, 0.0009555
3: -0.0049928, -0.0029961, -0.0049669, -0.0033775, -0.0009025, 0.0012830
4: -0.0005368, 0.0013680, -0.0003806, 0.0013400, -0.0010974, 0.0009770
5: 0.0027238, 0.0045980, 0.0028453, 0.0044735, -0.0010731, 0.0010124
6: -0.0112146, -0.0040566, -0.0110110, -0.0045506, -0.0037950, 0.0040169
7: 0.0026825, 0.0125827, 0.0036408, 0.0124393, -0.0057901, 0.0049963
8: 0.9913046, 0.9981083, 0.9917785, 0.9979764, -0.0038537, 0.0035562
9: -0.0141421, -0.0079019, -0.0140504, -0.0084244, -0.0031948, 0.0036016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024242, upper bound: 0.0021588
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024249, upper bound: 0.0021636
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0059188, 0.0092328, 0.0060066, 0.0090472, -0.0018208, 0.0021183
1: 0.0018360, 0.0026121, 0.0020529, 0.0026119, -0.0006238, 0.0003601
2: 0.0091901, 0.0110875, 0.0093370, 0.0110390, -0.0013121, 0.0010479
3: -0.0049338, -0.0028988, -0.0049331, -0.0031358, -0.0009512, 0.0012806
4: -0.0005891, 0.0013041, -0.0005170, 0.0013035, -0.0010444, 0.0008946
5: 0.0027531, 0.0046418, 0.0028288, 0.0045904, -0.0011111, 0.0010316
6: -0.0110091, -0.0038829, -0.0109272, -0.0040870, -0.0035427, 0.0038718
7: 0.0023805, 0.0122562, 0.0028615, 0.0122527, -0.0055499, 0.0046604
8: 0.9911380, 0.9978882, 0.9913338, 0.9978622, -0.0036695, 0.0032416
9: -0.0139333, -0.0077283, -0.0139310, -0.0079723, -0.0029314, 0.0034307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024433, upper bound: 0.0022938
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024442, upper bound: 0.0023028
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0059549, 0.0091902, 0.0060503, 0.0090503, -0.0017730, 0.0020352
1: 0.0018858, 0.0026120, 0.0021305, 0.0026212, -0.0005799, 0.0002820
2: 0.0092265, 0.0110676, 0.0093460, 0.0110148, -0.0012467, 0.0009975
3: -0.0049336, -0.0029614, -0.0049697, -0.0032243, -0.0008952, 0.0012622
4: -0.0005629, 0.0013039, -0.0004836, 0.0013431, -0.0010710, 0.0009028
5: 0.0027701, 0.0046207, 0.0028169, 0.0045648, -0.0010820, 0.0010236
6: -0.0109913, -0.0039668, -0.0110512, -0.0041885, -0.0035645, 0.0039544
7: 0.0025383, 0.0122550, 0.0030730, 0.0124553, -0.0056587, 0.0046594
8: 0.9912184, 0.9978824, 0.9914311, 0.9979970, -0.0037700, 0.0032851
9: -0.0139325, -0.0078155, -0.0140606, -0.0080847, -0.0029552, 0.0035155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024433, upper bound: 0.0022842
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024442, upper bound: 0.0022917
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058733, 0.0095834, 0.0060066, 0.0090472, -0.0019707, 0.0027117
1: 0.0015602, 0.0026271, 0.0020529, 0.0026119, -0.0009370, 0.0003885
2: 0.0089217, 0.0111127, 0.0093370, 0.0110390, -0.0017416, 0.0011308
3: -0.0049931, -0.0026253, -0.0049331, -0.0031358, -0.0010639, 0.0016672
4: -0.0006387, 0.0013684, -0.0005170, 0.0013035, -0.0011644, 0.0010166
5: 0.0026060, 0.0046685, 0.0028288, 0.0045904, -0.0013711, 0.0011193
6: -0.0113260, -0.0037771, -0.0109272, -0.0040870, -0.0040834, 0.0042202
7: 0.0019863, 0.0125848, 0.0028615, 0.0122527, -0.0063495, 0.0052842
8: 0.9910365, 0.9981436, 0.9913338, 0.9978622, -0.0040037, 0.0037035
9: -0.0141434, -0.0075560, -0.0139310, -0.0079723, -0.0033303, 0.0038366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024990, upper bound: 0.0022938
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024994, upper bound: 0.0023025
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0059132, 0.0095199, 0.0060503, 0.0090503, -0.0019279, 0.0026105
1: 0.0016205, 0.0026270, 0.0021305, 0.0026212, -0.0008841, 0.0003104
2: 0.0089752, 0.0110906, 0.0093460, 0.0110148, -0.0016622, 0.0010832
3: -0.0049929, -0.0026952, -0.0049697, -0.0032243, -0.0010074, 0.0016398
4: -0.0006096, 0.0013681, -0.0004836, 0.0013431, -0.0011934, 0.0010244
5: 0.0026263, 0.0046451, 0.0028169, 0.0045648, -0.0013347, 0.0011143
6: -0.0113065, -0.0038699, -0.0110512, -0.0041885, -0.0041026, 0.0043144
7: 0.0021619, 0.0125834, 0.0030730, 0.0124553, -0.0064582, 0.0052810
8: 0.9911255, 0.9981369, 0.9914311, 0.9979970, -0.0041154, 0.0037456
9: -0.0141425, -0.0076530, -0.0140606, -0.0080847, -0.0033527, 0.0039284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024990, upper bound: 0.0022842
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024994, upper bound: 0.0022909
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0061508, 0.0089282, 0.0062076, 0.0092383, -0.0024079, 0.0017617
1: 0.0022109, 0.0026122, 0.0017942, 0.0026132, -0.0002600, 0.0007096
2: 0.0094237, 0.0109593, 0.0091725, 0.0109279, -0.0009740, 0.0014777
3: -0.0049340, -0.0033459, -0.0049381, -0.0029754, -0.0014338, 0.0010290
4: -0.0004149, 0.0013044, -0.0004185, 0.0013088, -0.0011140, 0.0011285
5: 0.0028789, 0.0045059, 0.0027570, 0.0044727, -0.0010320, 0.0012879
6: -0.0108776, -0.0044221, -0.0110090, -0.0045541, -0.0040946, 0.0043728
7: 0.0034658, 0.0122577, 0.0032011, 0.0122800, -0.0056964, 0.0060288
8: 0.9916552, 0.9978484, 0.9917819, 0.9978991, -0.0040649, 0.0039282
9: -0.0139342, -0.0083125, -0.0139485, -0.0082823, -0.0037087, 0.0036425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023008, upper bound: 0.0022386
time: 0.80 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023077, upper bound: 0.0022399
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0062061, 0.0089856, 0.0062431, 0.0091892, -0.0023246, 0.0018133
1: 0.0022189, 0.0026205, 0.0018513, 0.0026131, -0.0002624, 0.0006586
2: 0.0093920, 0.0109287, 0.0092156, 0.0109082, -0.0010025, 0.0014070
3: -0.0049669, -0.0033775, -0.0049378, -0.0030451, -0.0014081, 0.0010387
4: -0.0003806, 0.0013400, -0.0003918, 0.0013085, -0.0011245, 0.0011562
5: 0.0028453, 0.0044735, 0.0027747, 0.0044519, -0.0010622, 0.0012585
6: -0.0110110, -0.0045506, -0.0109933, -0.0046367, -0.0042145, 0.0043834
7: 0.0036408, 0.0124393, 0.0033644, 0.0122786, -0.0057502, 0.0061359
8: 0.9917785, 0.9979764, 0.9918611, 0.9978941, -0.0040958, 0.0040432
9: -0.0140504, -0.0084244, -0.0139476, -0.0083717, -0.0037969, 0.0036768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022950, upper bound: 0.0022386
time: 0.82 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023022, upper bound: 0.0022399
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0061085, 0.0090259, 0.0062076, 0.0092383, -0.0024286, 0.0016550
1: 0.0022048, 0.0026263, 0.0017942, 0.0026132, -0.0002410, 0.0006960
2: 0.0093697, 0.0109826, 0.0091725, 0.0109279, -0.0009150, 0.0015024
3: -0.0049899, -0.0033217, -0.0049381, -0.0029754, -0.0013649, 0.0009540
4: -0.0004410, 0.0013649, -0.0004185, 0.0013088, -0.0010328, 0.0010612
5: 0.0028217, 0.0045307, 0.0027570, 0.0044727, -0.0009695, 0.0012774
6: -0.0111046, -0.0043239, -0.0110090, -0.0045541, -0.0038466, 0.0041184
7: 0.0033321, 0.0125668, 0.0032011, 0.0122800, -0.0052813, 0.0056680
8: 0.9915611, 0.9980661, 0.9917819, 0.9978991, -0.0037874, 0.0036903
9: -0.0141319, -0.0082270, -0.0139485, -0.0082823, -0.0034877, 0.0033770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023594, upper bound: 0.0022216
time: 0.78 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023702, upper bound: 0.0022216
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0061687, 0.0090854, 0.0062431, 0.0091892, -0.0023324, 0.0017081
1: 0.0022135, 0.0026349, 0.0018513, 0.0026131, -0.0002437, 0.0006442
2: 0.0093368, 0.0109494, 0.0092156, 0.0109082, -0.0009444, 0.0014256
3: -0.0050239, -0.0033561, -0.0049378, -0.0030451, -0.0013370, 0.0009644
4: -0.0004038, 0.0014017, -0.0003918, 0.0013085, -0.0010440, 0.0010888
5: 0.0027869, 0.0044955, 0.0027747, 0.0044519, -0.0010006, 0.0012414
6: -0.0112429, -0.0044636, -0.0109933, -0.0046367, -0.0039702, 0.0041254
7: 0.0035224, 0.0127551, 0.0033644, 0.0122786, -0.0053386, 0.0057722
8: 0.9916951, 0.9981989, 0.9918611, 0.9978941, -0.0038189, 0.0038088
9: -0.0142523, -0.0083487, -0.0139476, -0.0083717, -0.0035754, 0.0034136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023553, upper bound: 0.0022216
time: 0.83 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023655, upper bound: 0.0022216
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0059614, 0.0090903, 0.0061790, 0.0094991, -0.0030423, 0.0020977
1: 0.0019981, 0.0026120, 0.0015568, 0.0026131, -0.0004692, 0.0009740
2: 0.0093044, 0.0110640, 0.0089506, 0.0109437, -0.0012187, 0.0019371
3: -0.0049335, -0.0030681, -0.0049379, -0.0027326, -0.0017276, 0.0012586
4: -0.0005489, 0.0013038, -0.0004564, 0.0013086, -0.0011936, 0.0011951
5: 0.0028105, 0.0046169, 0.0026705, 0.0044894, -0.0011781, 0.0015563
6: -0.0109481, -0.0039819, -0.0110703, -0.0044877, -0.0043618, 0.0047426
7: 0.0026718, 0.0122546, 0.0028695, 0.0122788, -0.0062104, 0.0065267
8: 0.9912329, 0.9978694, 0.9917182, 0.9979175, -0.0043290, 0.0041183
9: -0.0139322, -0.0078662, -0.0139477, -0.0081484, -0.0039384, 0.0039108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024267, upper bound: 0.0023015
time: 0.85 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024269, upper bound: 0.0023159
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0059166, 0.0093975, 0.0061790, 0.0094991, -0.0031269, 0.0024836
1: 0.0017375, 0.0026270, 0.0015568, 0.0026131, -0.0007254, 0.0009613
2: 0.0090859, 0.0110887, 0.0089506, 0.0109437, -0.0015339, 0.0019905
3: -0.0049928, -0.0028107, -0.0049379, -0.0027326, -0.0016634, 0.0014234
4: -0.0005977, 0.0013680, -0.0004564, 0.0013086, -0.0011290, 0.0011355
5: 0.0026679, 0.0046431, 0.0026705, 0.0044894, -0.0013164, 0.0015865
6: -0.0112675, -0.0038779, -0.0110703, -0.0044877, -0.0042901, 0.0045533
7: 0.0022888, 0.0125828, 0.0028695, 0.0122788, -0.0060209, 0.0061877
8: 0.9911331, 0.9981245, 0.9917182, 0.9979175, -0.0040496, 0.0039567
9: -0.0141421, -0.0076967, -0.0139477, -0.0081484, -0.0037408, 0.0037100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024267, upper bound: 0.0022999
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024269, upper bound: 0.0023075
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0059966, 0.0090498, 0.0062225, 0.0094302, -0.0028719, 0.0020399
1: 0.0020503, 0.0026120, 0.0016663, 0.0026224, -0.0004274, 0.0008682
2: 0.0093352, 0.0110445, 0.0090228, 0.0109196, -0.0011680, 0.0017908
3: -0.0049332, -0.0031293, -0.0049747, -0.0028537, -0.0016346, 0.0012516
4: -0.0005231, 0.0013036, -0.0004216, 0.0013484, -0.0012295, 0.0011989
5: 0.0028277, 0.0045962, 0.0026805, 0.0044639, -0.0011610, 0.0015031
6: -0.0109287, -0.0040638, -0.0111770, -0.0045889, -0.0043837, 0.0048334
7: 0.0028309, 0.0122533, 0.0031022, 0.0124826, -0.0063675, 0.0064789
8: 0.9913116, 0.9978630, 0.9918153, 0.9980462, -0.0044572, 0.0041562
9: -0.0139314, -0.0079524, -0.0140781, -0.0082661, -0.0039457, 0.0040261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024267, upper bound: 0.0023043
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024269, upper bound: 0.0023155
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0059561, 0.0093499, 0.0062225, 0.0094302, -0.0029315, 0.0024063
1: 0.0017961, 0.0026269, 0.0016663, 0.0026224, -0.0006765, 0.0008552
2: 0.0091267, 0.0110669, 0.0090228, 0.0109196, -0.0014651, 0.0018404
3: -0.0049925, -0.0028808, -0.0049747, -0.0028537, -0.0015695, 0.0014062
4: -0.0005691, 0.0013677, -0.0004216, 0.0013484, -0.0011654, 0.0011445
5: 0.0026871, 0.0046200, 0.0026805, 0.0044639, -0.0012895, 0.0015149
6: -0.0112481, -0.0039695, -0.0111770, -0.0045889, -0.0043249, 0.0046180
7: 0.0024636, 0.0125813, 0.0031022, 0.0124826, -0.0061761, 0.0061704
8: 0.9912211, 0.9981179, 0.9918153, 0.9980462, -0.0041750, 0.0040129
9: -0.0141412, -0.0077922, -0.0140781, -0.0082661, -0.0037661, 0.0038266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024267, upper bound: 0.0022918
time: 0.76 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024269, upper bound: 0.0022985
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0059961, 0.0090005, 0.0061085, 0.0090259, -0.0016956, 0.0016483
1: 0.0021062, 0.0026121, 0.0022048, 0.0026263, -0.0003351, 0.0002218
2: 0.0093711, 0.0110448, 0.0093697, 0.0109826, -0.0009360, 0.0009375
3: -0.0049337, -0.0031801, -0.0049899, -0.0033217, -0.0008778, 0.0010514
4: -0.0005185, 0.0013040, -0.0004410, 0.0013649, -0.0010577, 0.0009503
5: 0.0028484, 0.0045966, 0.0028217, 0.0045307, -0.0009444, 0.0009933
6: -0.0109089, -0.0040625, -0.0111046, -0.0043239, -0.0036121, 0.0039411
7: 0.0028862, 0.0122556, 0.0033321, 0.0125668, -0.0054609, 0.0048593
8: 0.9913104, 0.9978576, 0.9915611, 0.9980661, -0.0037809, 0.0034374
9: -0.0139329, -0.0079700, -0.0141319, -0.0082270, -0.0031072, 0.0034625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021973
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0022046
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0059536, 0.0093088, 0.0061085, 0.0090259, -0.0015460, 0.0020429
1: 0.0018427, 0.0026271, 0.0022048, 0.0026263, -0.0006031, 0.0002069
2: 0.0091595, 0.0110683, 0.0093697, 0.0109826, -0.0012593, 0.0008547
3: -0.0049930, -0.0029249, -0.0049899, -0.0033217, -0.0008190, 0.0012288
4: -0.0005660, 0.0013683, -0.0004410, 0.0013649, -0.0009878, 0.0008866
5: 0.0027036, 0.0046214, 0.0028217, 0.0045307, -0.0010786, 0.0009056
6: -0.0112341, -0.0039638, -0.0111046, -0.0043239, -0.0035239, 0.0035932
7: 0.0025050, 0.0125841, 0.0033321, 0.0125668, -0.0052549, 0.0045341
8: 0.9912156, 0.9981148, 0.9915611, 0.9980661, -0.0034472, 0.0032504
9: -0.0141430, -0.0078043, -0.0141319, -0.0082270, -0.0028992, 0.0032450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021929
time: 0.82 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021999
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0060329, 0.0089664, 0.0061687, 0.0090854, -0.0017678, 0.0015972
1: 0.0021504, 0.0026120, 0.0022135, 0.0026349, -0.0003018, 0.0002228
2: 0.0093957, 0.0110244, 0.0093368, 0.0109494, -0.0008942, 0.0009774
3: -0.0049335, -0.0032356, -0.0050239, -0.0033561, -0.0008820, 0.0010546
4: -0.0004922, 0.0013038, -0.0004038, 0.0014017, -0.0010987, 0.0009548
5: 0.0028630, 0.0045750, 0.0027869, 0.0044955, -0.0009255, 0.0010356
6: -0.0108933, -0.0041481, -0.0112429, -0.0044636, -0.0036092, 0.0041090
7: 0.0030428, 0.0122544, 0.0035224, 0.0127551, -0.0056460, 0.0048826
8: 0.9913924, 0.9978520, 0.9916951, 0.9981989, -0.0039420, 0.0034473
9: -0.0139321, -0.0080576, -0.0142523, -0.0083487, -0.0031220, 0.0035946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0022012
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0022068
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0059935, 0.0092612, 0.0061687, 0.0090854, -0.0016234, 0.0019378
1: 0.0019031, 0.0026270, 0.0022135, 0.0026349, -0.0005513, 0.0002092
2: 0.0091964, 0.0110462, 0.0093368, 0.0109494, -0.0011787, 0.0008975
3: -0.0049928, -0.0029961, -0.0050239, -0.0033561, -0.0008279, 0.0012114
4: -0.0005368, 0.0013680, -0.0004038, 0.0014017, -0.0010305, 0.0008962
5: 0.0027238, 0.0045980, 0.0027869, 0.0044955, -0.0010400, 0.0009510
6: -0.0112146, -0.0040566, -0.0112429, -0.0044636, -0.0035233, 0.0037732
7: 0.0026825, 0.0125827, 0.0035224, 0.0127551, -0.0054344, 0.0045830
8: 0.9913046, 0.9981083, 0.9916951, 0.9981989, -0.0036199, 0.0032757
9: -0.0141421, -0.0079019, -0.0142523, -0.0083487, -0.0029305, 0.0033818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021907
time: 0.76 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021960
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058306, 0.0096205, 0.0060425, 0.0091640, -0.0021481, 0.0029181
1: 0.0014560, 0.0026112, 0.0020043, 0.0026243, -0.0010635, 0.0004142
2: 0.0088500, 0.0111363, 0.0092645, 0.0110191, -0.0019035, 0.0012369
3: -0.0049301, -0.0025186, -0.0049820, -0.0031107, -0.0009766, 0.0018485
4: -0.0006726, 0.0013001, -0.0004990, 0.0013563, -0.0012684, 0.0008673
5: 0.0026127, 0.0046935, 0.0027669, 0.0045694, -0.0014356, 0.0012201
6: -0.0111200, -0.0036779, -0.0111391, -0.0041705, -0.0037189, 0.0045912
7: 0.0017641, 0.0122356, 0.0029267, 0.0125231, -0.0069280, 0.0045541
8: 0.9909413, 0.9979134, 0.9914139, 0.9980567, -0.0043528, 0.0032000
9: -0.0139202, -0.0074411, -0.0141039, -0.0080291, -0.0028447, 0.0041796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024361, upper bound: 0.0023326
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024361, upper bound: 0.0023374
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058251, 0.0096495, 0.0060258, 0.0092220, -0.0022438, 0.0029909
1: 0.0014338, 0.0026115, 0.0019526, 0.0026279, -0.0010910, 0.0004723
2: 0.0088235, 0.0111393, 0.0092252, 0.0110284, -0.0019519, 0.0013076
3: -0.0049316, -0.0024976, -0.0049965, -0.0030558, -0.0010395, 0.0018947
4: -0.0006776, 0.0013017, -0.0005138, 0.0013720, -0.0012930, 0.0008861
5: 0.0026015, 0.0046967, 0.0027377, 0.0045792, -0.0014702, 0.0012621
6: -0.0111337, -0.0036651, -0.0112100, -0.0041315, -0.0037907, 0.0047045
7: 0.0017291, 0.0122439, 0.0028232, 0.0126033, -0.0070626, 0.0046783
8: 0.9909291, 0.9979216, 0.9913765, 0.9981170, -0.0044467, 0.0032562
9: -0.0139254, -0.0074244, -0.0141553, -0.0079787, -0.0029079, 0.0042604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024370, upper bound: 0.0023478
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024370, upper bound: 0.0023458
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058676, 0.0095516, 0.0060936, 0.0091527, -0.0021196, 0.0028038
1: 0.0015198, 0.0026111, 0.0020994, 0.0026328, -0.0010117, 0.0003154
2: 0.0089151, 0.0111158, 0.0092853, 0.0109909, -0.0018177, 0.0011910
3: -0.0049299, -0.0025907, -0.0050157, -0.0032168, -0.0008988, 0.0018278
4: -0.0006445, 0.0012999, -0.0004600, 0.0013928, -0.0013098, 0.0008723
5: 0.0026383, 0.0046718, 0.0027607, 0.0045394, -0.0013903, 0.0012241
6: -0.0110997, -0.0037640, -0.0112464, -0.0042892, -0.0037152, 0.0047327
7: 0.0019383, 0.0122347, 0.0031732, 0.0127096, -0.0070978, 0.0045236
8: 0.9910240, 0.9979065, 0.9915277, 0.9981788, -0.0045125, 0.0032351
9: -0.0139195, -0.0075352, -0.0142232, -0.0081602, -0.0028569, 0.0043118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024361, upper bound: 0.0023359
time: 0.79 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024361, upper bound: 0.0023292
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058632, 0.0095777, 0.0060818, 0.0092029, -0.0021872, 0.0028745
1: 0.0014992, 0.0026115, 0.0020552, 0.0026363, -0.0010373, 0.0003666
2: 0.0088922, 0.0111182, 0.0092520, 0.0109974, -0.0018640, 0.0012387
3: -0.0049314, -0.0025703, -0.0050297, -0.0031714, -0.0009535, 0.0018708
4: -0.0006492, 0.0013015, -0.0004709, 0.0014080, -0.0013312, 0.0008911
5: 0.0026280, 0.0046744, 0.0027359, 0.0045464, -0.0014241, 0.0012574
6: -0.0111128, -0.0037538, -0.0113132, -0.0042617, -0.0037878, 0.0048217
7: 0.0019049, 0.0122429, 0.0030932, 0.0127872, -0.0072234, 0.0046436
8: 0.9910140, 0.9979143, 0.9915015, 0.9982367, -0.0045890, 0.0032932
9: -0.0139248, -0.0075194, -0.0142728, -0.0081227, -0.0029202, 0.0043835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024370, upper bound: 0.0023464
time: 0.78 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024370, upper bound: 0.0023379
time: 0.83 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.58 seconds
IS_A1_B1_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022073, upper bound: 0.0021476
IS_A1_B1_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022094, upper bound: 0.0021522
IS_A1_B1_B1_B1_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021928, upper bound: 0.0021495
IS_A1_B1_B1_B1_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022038, upper bound: 0.0021522
IS_A1_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022531, upper bound: 0.0021470
IS_A1_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022550, upper bound: 0.0021518
IS_A1_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022563, upper bound: 0.0021470
IS_A1_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022579, upper bound: 0.0021518
IS_A1_B1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022753, upper bound: 0.0022760
IS_A1_B1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022771, upper bound: 0.0022874
IS_A1_B1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022753, upper bound: 0.0022673
IS_A1_B1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022771, upper bound: 0.0022771
IS_A1_B1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023314, upper bound: 0.0022754
IS_A1_B1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023332, upper bound: 0.0022856
IS_A1_B1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023314, upper bound: 0.0022656
IS_A1_B1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023332, upper bound: 0.0022746
IS_A1_B1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021917, upper bound: 0.0023029
IS_A1_B1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021933, upper bound: 0.0023096
IS_A1_B1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021917, upper bound: 0.0022954
IS_A1_B1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021933, upper bound: 0.0023034
IS_A1_B1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022386, upper bound: 0.0023008
IS_A1_B1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022399, upper bound: 0.0023077
IS_A1_B1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022386, upper bound: 0.0022951
IS_A1_B1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022399, upper bound: 0.0023022
IS_A1_B1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022673, upper bound: 0.0024272
IS_A1_B1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022698, upper bound: 0.0024379
IS_A1_B1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022673, upper bound: 0.0024249
IS_A1_B1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022698, upper bound: 0.0024350
IS_A1_B1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023139, upper bound: 0.0024223
IS_A1_B1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023155, upper bound: 0.0024297
IS_A1_B1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023139, upper bound: 0.0024189
IS_A1_B1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023155, upper bound: 0.0024269
IS_A1_B2_B1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021972, upper bound: 0.0021837
IS_A1_B2_B1_B1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021972, upper bound: 0.0021745
IS_A1_B2_B1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021989, upper bound: 0.0021933
IS_A1_B2_B1_B1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021989, upper bound: 0.0021804
IS_A1_B2_B1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021972, upper bound: 0.0021887
IS_A1_B2_B1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021972, upper bound: 0.0021731
IS_A1_B2_B1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021989, upper bound: 0.0021964
IS_A1_B2_B1_B1_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021989, upper bound: 0.0021757
IS_A1_B2_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022721, upper bound: 0.0023139
IS_A1_B2_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022721, upper bound: 0.0023088
IS_A1_B2_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022746, upper bound: 0.0023316
IS_A1_B2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022746, upper bound: 0.0023150
IS_A1_B2_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022721, upper bound: 0.0023165
IS_A1_B2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022721, upper bound: 0.0022978
IS_A1_B2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022746, upper bound: 0.0023332
IS_A1_B2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022746, upper bound: 0.0023041
IS_A1_B2_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021906, upper bound: 0.0023359
IS_A1_B2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021906, upper bound: 0.0023299
IS_A1_B2_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021925, upper bound: 0.0023489
IS_A1_B2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021925, upper bound: 0.0023327
IS_A1_B2_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021906, upper bound: 0.0023325
IS_A1_B2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021906, upper bound: 0.0023231
IS_A1_B2_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021925, upper bound: 0.0023455
IS_A1_B2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0021925, upper bound: 0.0023267
IS_A1_B2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022667, upper bound: 0.0024747
IS_A1_B2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022667, upper bound: 0.0024512
IS_A1_B2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022692, upper bound: 0.0024946
IS_A1_B2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022692, upper bound: 0.0024595
IS_A1_B2_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022667, upper bound: 0.0024723
IS_A1_B2_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022667, upper bound: 0.0024504
IS_A1_B2_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022692, upper bound: 0.0024865
IS_A1_B2_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022692, upper bound: 0.0024565
IS_A2_B1_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023029, upper bound: 0.0021917
IS_A2_B1_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023096, upper bound: 0.0021933
IS_A2_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022953, upper bound: 0.0021917
IS_A2_B1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023034, upper bound: 0.0021933
IS_A2_B1_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023358, upper bound: 0.0021906
IS_A2_B1_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023488, upper bound: 0.0021925
IS_A2_B1_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023325, upper bound: 0.0021906
IS_A2_B1_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023455, upper bound: 0.0021925
IS_A2_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024272, upper bound: 0.0022673
IS_A2_B1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024379, upper bound: 0.0022699
IS_A2_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024249, upper bound: 0.0022673
IS_A2_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024349, upper bound: 0.0022699
IS_A2_B1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024747, upper bound: 0.0022666
IS_A2_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024946, upper bound: 0.0022693
IS_A2_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024723, upper bound: 0.0022666
IS_A2_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024866, upper bound: 0.0022693
IS_A2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023606, upper bound: 0.0021626
IS_A2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023741, upper bound: 0.0021636
IS_A2_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023628, upper bound: 0.0021626
IS_A2_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023727, upper bound: 0.0021636
IS_A2_B1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024242, upper bound: 0.0021605
IS_A2_B1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024249, upper bound: 0.0021670
IS_A2_B1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024242, upper bound: 0.0021588
IS_A2_B1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024249, upper bound: 0.0021636
IS_A2_B1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024433, upper bound: 0.0022938
IS_A2_B1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024442, upper bound: 0.0023028
IS_A2_B1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024433, upper bound: 0.0022842
IS_A2_B1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024442, upper bound: 0.0022917
IS_A2_B1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024990, upper bound: 0.0022938
IS_A2_B1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024994, upper bound: 0.0023025
IS_A2_B1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024990, upper bound: 0.0022842
IS_A2_B1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024994, upper bound: 0.0022909
IS_A2_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023008, upper bound: 0.0022386
IS_A2_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023077, upper bound: 0.0022399
IS_A2_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0022950, upper bound: 0.0022386
IS_A2_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023022, upper bound: 0.0022399
IS_A2_B2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023594, upper bound: 0.0022216
IS_A2_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023702, upper bound: 0.0022216
IS_A2_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023553, upper bound: 0.0022216
IS_A2_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023655, upper bound: 0.0022216
IS_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024267, upper bound: 0.0023015
IS_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024269, upper bound: 0.0023159
IS_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024267, upper bound: 0.0022999
IS_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024269, upper bound: 0.0023075
IS_A2_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024267, upper bound: 0.0023043
IS_A2_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024269, upper bound: 0.0023155
IS_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024267, upper bound: 0.0022918
IS_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024269, upper bound: 0.0022985
IS_A2_B2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021973
IS_A2_B2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0022046
IS_A2_B2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021929
IS_A2_B2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021999
IS_A2_B2_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0022012
IS_A2_B2_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0022068
IS_A2_B2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021907
IS_A2_B2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0023642, upper bound: 0.0021960
IS_A2_B2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024361, upper bound: 0.0023326
IS_A2_B2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024361, upper bound: 0.0023374
IS_A2_B2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024370, upper bound: 0.0023478
IS_A2_B2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024370, upper bound: 0.0023458
IS_A2_B2_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024361, upper bound: 0.0023359
IS_A2_B2_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024361, upper bound: 0.0023292
IS_A2_B2_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024370, upper bound: 0.0023464
IS_A2_B2_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.58
Output dim: 8, lower bound: -0.0024370, upper bound: 0.0023379

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062832, 0.0091123, 0.0064312, 0.0088170, -0.0014622, 0.0017919
1: 0.0019421, 0.0026120, 0.0022514, 0.0025961, -0.0005188, 0.0002067
2: 0.0092813, 0.0108861, 0.0094852, 0.0108042, -0.0010754, 0.0008084
3: -0.0049332, -0.0031513, -0.0048705, -0.0035063, -0.0008182, 0.0011254
4: -0.0003594, 0.0013035, -0.0002412, 0.0012356, -0.0009316, 0.0008858
5: 0.0028054, 0.0044284, 0.0029441, 0.0043416, -0.0009788, 0.0008566
6: -0.0109508, -0.0047298, -0.0106192, -0.0050740, -0.0034463, 0.0033986
7: 0.0035799, 0.0122530, 0.0043536, 0.0119057, -0.0049382, 0.0045297
8: 0.9919504, 0.9978691, 0.9922807, 0.9976005, -0.0032605, 0.0032254
9: -0.0139313, -0.0084812, -0.0137092, -0.0088802, -0.0028964, 0.0030588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020968, upper bound: 0.0019648
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020869, upper bound: 0.0019648
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062809, 0.0091304, 0.0064154, 0.0088373, -0.0015118, 0.0018423
1: 0.0019222, 0.0026123, 0.0022491, 0.0025990, -0.0005465, 0.0002095
2: 0.0092668, 0.0108873, 0.0094740, 0.0108130, -0.0011106, 0.0008358
3: -0.0049347, -0.0031321, -0.0048820, -0.0034972, -0.0008293, 0.0011730
4: -0.0003624, 0.0013052, -0.0002510, 0.0012481, -0.0009639, 0.0008978
5: 0.0027980, 0.0044297, 0.0029322, 0.0043509, -0.0010013, 0.0008856
6: -0.0109624, -0.0047246, -0.0106663, -0.0050372, -0.0035005, 0.0035138
7: 0.0035536, 0.0122614, 0.0043035, 0.0119698, -0.0051197, 0.0045910
8: 0.9919454, 0.9978766, 0.9922453, 0.9976457, -0.0033710, 0.0032708
9: -0.0139367, -0.0084706, -0.0137502, -0.0088481, -0.0029356, 0.0031656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020968, upper bound: 0.0019688
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020869, upper bound: 0.0019689
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063343, 0.0091064, 0.0064668, 0.0088168, -0.0014759, 0.0017429
1: 0.0020347, 0.0026212, 0.0022566, 0.0025961, -0.0004279, 0.0002183
2: 0.0093009, 0.0108578, 0.0094853, 0.0107845, -0.0010123, 0.0008160
3: -0.0049700, -0.0032576, -0.0048703, -0.0035266, -0.0008641, 0.0010463
4: -0.0003204, 0.0013433, -0.0002192, 0.0012355, -0.0009326, 0.0009354
5: 0.0027956, 0.0043984, 0.0029442, 0.0043208, -0.0009810, 0.0008646
6: -0.0110698, -0.0048487, -0.0106187, -0.0051566, -0.0036076, 0.0034304
7: 0.0038257, 0.0124565, 0.0044661, 0.0119050, -0.0048900, 0.0047833
8: 0.9920645, 0.9980031, 0.9923598, 0.9976000, -0.0032909, 0.0033978
9: -0.0140614, -0.0086123, -0.0137087, -0.0089521, -0.0030586, 0.0030585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020956, upper bound: 0.0019648
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020816, upper bound: 0.0019644
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063325, 0.0091190, 0.0064550, 0.0088370, -0.0015256, 0.0017760
1: 0.0020193, 0.0026216, 0.0022549, 0.0025990, -0.0004512, 0.0002203
2: 0.0092912, 0.0108588, 0.0094741, 0.0107911, -0.0010371, 0.0008435
3: -0.0049715, -0.0032428, -0.0048819, -0.0035198, -0.0008719, 0.0010900
4: -0.0003226, 0.0013450, -0.0002265, 0.0012480, -0.0009648, 0.0009439
5: 0.0027900, 0.0043995, 0.0029323, 0.0043277, -0.0009965, 0.0008937
6: -0.0110798, -0.0048446, -0.0106657, -0.0051292, -0.0036472, 0.0035460
7: 0.0038074, 0.0124651, 0.0044289, 0.0119691, -0.0050700, 0.0048269
8: 0.9920606, 0.9980101, 0.9923337, 0.9976451, -0.0034019, 0.0034309
9: -0.0140669, -0.0086045, -0.0137497, -0.0089283, -0.0030864, 0.0031647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020956, upper bound: 0.0019688
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020816, upper bound: 0.0019688
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0062157, 0.0090716, 0.0063493, 0.0088284, -0.0014995, 0.0018536
1: 0.0018943, 0.0025979, 0.0022267, 0.0025966, -0.0005668, 0.0001899
2: 0.0092870, 0.0109233, 0.0094776, 0.0108495, -0.0011523, 0.0008312
3: -0.0048775, -0.0030724, -0.0048725, -0.0034464, -0.0007162, 0.0011625
4: -0.0004051, 0.0012432, -0.0002933, 0.0012378, -0.0009493, 0.0007638
5: 0.0028407, 0.0044679, 0.0029386, 0.0043896, -0.0009698, 0.0008764
6: -0.0107408, -0.0045731, -0.0106312, -0.0048835, -0.0030760, 0.0034618
7: 0.0033224, 0.0119444, 0.0040792, 0.0119168, -0.0050442, 0.0039135
8: 0.9918001, 0.9976574, 0.9920980, 0.9976096, -0.0033173, 0.0028075
9: -0.0137339, -0.0083299, -0.0137163, -0.0087094, -0.0024982, 0.0031180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021098, upper bound: 0.0021368
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021071, upper bound: 0.0021216
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0062136, 0.0090913, 0.0063336, 0.0088812, -0.0016094, 0.0019188
1: 0.0018747, 0.0025982, 0.0021684, 0.0025992, -0.0005937, 0.0002505
2: 0.0092706, 0.0109245, 0.0094417, 0.0108582, -0.0011969, 0.0009044
3: -0.0048789, -0.0030540, -0.0048827, -0.0033831, -0.0007820, 0.0012091
4: -0.0004077, 0.0012447, -0.0003085, 0.0012489, -0.0009792, 0.0007834
5: 0.0028331, 0.0044691, 0.0029134, 0.0043989, -0.0009999, 0.0009294
6: -0.0107522, -0.0045681, -0.0106878, -0.0048470, -0.0031432, 0.0035926
7: 0.0032970, 0.0119525, 0.0039665, 0.0119736, -0.0052051, 0.0040462
8: 0.9917952, 0.9976647, 0.9920628, 0.9976545, -0.0034267, 0.0028632
9: -0.0137391, -0.0083206, -0.0137526, -0.0086571, -0.0025644, 0.0032167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021296, upper bound: 0.0021256
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021071, upper bound: 0.0021260
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0062542, 0.0090278, 0.0064004, 0.0088739, -0.0015342, 0.0017667
1: 0.0019477, 0.0025978, 0.0022470, 0.0026043, -0.0005198, 0.0001813
2: 0.0093237, 0.0109021, 0.0094537, 0.0108212, -0.0010832, 0.0008482
3: -0.0048773, -0.0031407, -0.0049030, -0.0034886, -0.0007176, 0.0011394
4: -0.0003769, 0.0012430, -0.0002603, 0.0012708, -0.0009739, 0.0007769
5: 0.0028566, 0.0044453, 0.0029107, 0.0043597, -0.0009395, 0.0008987
6: -0.0107257, -0.0046626, -0.0107514, -0.0050024, -0.0030956, 0.0035659
7: 0.0034945, 0.0119435, 0.0042561, 0.0120858, -0.0051412, 0.0039727
8: 0.9918859, 0.9976525, 0.9922120, 0.9977273, -0.0034210, 0.0028497
9: -0.0137333, -0.0084243, -0.0138243, -0.0088178, -0.0025403, 0.0031961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021098, upper bound: 0.0021226
time: 0.77 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021071, upper bound: 0.0021004
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0090463, 0.0063839, 0.0088957, -0.0015729, 0.0018296
1: 0.0019292, 0.0025982, 0.0022446, 0.0026075, -0.0005459, 0.0001847
2: 0.0093093, 0.0109035, 0.0094416, 0.0108304, -0.0011261, 0.0008696
3: -0.0048787, -0.0031231, -0.0049155, -0.0034792, -0.0007311, 0.0011824
4: -0.0003799, 0.0012446, -0.0002705, 0.0012843, -0.0010000, 0.0007914
5: 0.0028491, 0.0044469, 0.0028979, 0.0043694, -0.0009689, 0.0009214
6: -0.0107369, -0.0046564, -0.0108022, -0.0049640, -0.0031625, 0.0036559
7: 0.0034672, 0.0119515, 0.0042038, 0.0121549, -0.0052837, 0.0040472
8: 0.9918801, 0.9976598, 0.9921752, 0.9977760, -0.0035073, 0.0029053
9: -0.0137384, -0.0084135, -0.0138685, -0.0087844, -0.0025879, 0.0032825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021098, upper bound: 0.0021296
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021071, upper bound: 0.0021071
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061681, 0.0094527, 0.0063493, 0.0088284, -0.0016514, 0.0025008
1: 0.0015885, 0.0026121, 0.0022267, 0.0025966, -0.0009101, 0.0002181
2: 0.0089859, 0.0109497, 0.0094776, 0.0108495, -0.0016243, 0.0009151
3: -0.0049336, -0.0027566, -0.0048725, -0.0034464, -0.0008281, 0.0015778
4: -0.0004614, 0.0013039, -0.0002933, 0.0012378, -0.0010723, 0.0008850
5: 0.0026883, 0.0044958, 0.0029386, 0.0043896, -0.0012517, 0.0009653
6: -0.0110443, -0.0044623, -0.0106312, -0.0048835, -0.0036269, 0.0038147
7: 0.0028614, 0.0122552, 0.0040792, 0.0119168, -0.0058759, 0.0045330
8: 0.9916939, 0.9978982, 0.9920980, 0.9976096, -0.0036558, 0.0032706
9: -0.0139326, -0.0081329, -0.0137163, -0.0087094, -0.0028943, 0.0035351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021868, upper bound: 0.0021239
time: 0.84 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021717, upper bound: 0.0021216
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0061632, 0.0094771, 0.0063336, 0.0088812, -0.0017564, 0.0025636
1: 0.0015699, 0.0026124, 0.0021684, 0.0025992, -0.0009342, 0.0002788
2: 0.0089673, 0.0109524, 0.0094417, 0.0108582, -0.0016667, 0.0009856
3: -0.0049351, -0.0027369, -0.0048827, -0.0033831, -0.0008942, 0.0016176
4: -0.0004650, 0.0013056, -0.0003085, 0.0012489, -0.0010981, 0.0009048
5: 0.0026791, 0.0044986, 0.0029134, 0.0043989, -0.0012808, 0.0010155
6: -0.0110556, -0.0044510, -0.0106878, -0.0048470, -0.0036923, 0.0039341
7: 0.0028337, 0.0122638, 0.0039665, 0.0119736, -0.0060119, 0.0046674
8: 0.9916830, 0.9979059, 0.9920628, 0.9976545, -0.0037543, 0.0033265
9: -0.0139381, -0.0081209, -0.0137526, -0.0086571, -0.0029617, 0.0036192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021868, upper bound: 0.0021246
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021717, upper bound: 0.0021249
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0062023, 0.0093841, 0.0064004, 0.0088739, -0.0016919, 0.0023957
1: 0.0016514, 0.0026120, 0.0022470, 0.0026043, -0.0008540, 0.0002095
2: 0.0090434, 0.0109308, 0.0094537, 0.0108212, -0.0015407, 0.0009354
3: -0.0049334, -0.0028304, -0.0049030, -0.0034886, -0.0008293, 0.0015456
4: -0.0004353, 0.0013037, -0.0002603, 0.0012708, -0.0010998, 0.0008977
5: 0.0027117, 0.0044757, 0.0029107, 0.0043597, -0.0012134, 0.0009911
6: -0.0110280, -0.0045419, -0.0107514, -0.0050024, -0.0036448, 0.0039324
7: 0.0030279, 0.0122540, 0.0042561, 0.0120858, -0.0059737, 0.0045908
8: 0.9917702, 0.9978925, 0.9922120, 0.9977273, -0.0037726, 0.0033112
9: -0.0139319, -0.0082207, -0.0138243, -0.0088178, -0.0029355, 0.0036220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021758, upper bound: 0.0021222
time: 0.85 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021695, upper bound: 0.0020986
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0061983, 0.0094064, 0.0063839, 0.0088957, -0.0017235, 0.0024582
1: 0.0016337, 0.0026124, 0.0022446, 0.0026075, -0.0008777, 0.0002130
2: 0.0090248, 0.0109330, 0.0094416, 0.0108304, -0.0015830, 0.0009529
3: -0.0049349, -0.0028121, -0.0049155, -0.0034792, -0.0008430, 0.0015836
4: -0.0004389, 0.0013054, -0.0002705, 0.0012843, -0.0011213, 0.0009126
5: 0.0027032, 0.0044781, 0.0028979, 0.0043694, -0.0012421, 0.0010096
6: -0.0110391, -0.0045326, -0.0108022, -0.0049640, -0.0037101, 0.0040059
7: 0.0029994, 0.0122625, 0.0042038, 0.0121549, -0.0060951, 0.0046668
8: 0.9917613, 0.9979000, 0.9921752, 0.9977760, -0.0038431, 0.0033673
9: -0.0139373, -0.0082082, -0.0138685, -0.0087844, -0.0029841, 0.0036934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021758, upper bound: 0.0021268
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021695, upper bound: 0.0021046
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0062858, 0.0088599, 0.0062271, 0.0089093, -0.0015536, 0.0015905
1: 0.0021810, 0.0025979, 0.0022219, 0.0026094, -0.0002721, 0.0002215
2: 0.0094559, 0.0108846, 0.0094342, 0.0109171, -0.0008910, 0.0008590
3: -0.0048774, -0.0033739, -0.0049232, -0.0033895, -0.0008767, 0.0009337
4: -0.0003362, 0.0012431, -0.0003676, 0.0012927, -0.0009662, 0.0009491
5: 0.0029238, 0.0044268, 0.0028900, 0.0044612, -0.0009211, 0.0009101
6: -0.0106622, -0.0047359, -0.0108336, -0.0045995, -0.0035878, 0.0036110
7: 0.0038362, 0.0119441, 0.0037074, 0.0121978, -0.0049696, 0.0048532
8: 0.9919564, 0.9976324, 0.9918255, 0.9978063, -0.0034643, 0.0034267
9: -0.0137337, -0.0085672, -0.0138959, -0.0084670, -0.0031033, 0.0031615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020201, upper bound: 0.0021472
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020189, upper bound: 0.0021286
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0062840, 0.0088696, 0.0062119, 0.0089321, -0.0015928, 0.0016206
1: 0.0021703, 0.0025982, 0.0022197, 0.0026127, -0.0002895, 0.0002237
2: 0.0094493, 0.0108856, 0.0094215, 0.0109255, -0.0009111, 0.0008806
3: -0.0048788, -0.0033633, -0.0049363, -0.0033808, -0.0008855, 0.0009660
4: -0.0003384, 0.0012447, -0.0003770, 0.0013069, -0.0009914, 0.0009586
5: 0.0029194, 0.0044279, 0.0028766, 0.0044701, -0.0009361, 0.0009330
6: -0.0106712, -0.0047317, -0.0108867, -0.0045641, -0.0036288, 0.0037020
7: 0.0038191, 0.0119520, 0.0036592, 0.0122701, -0.0051044, 0.0049020
8: 0.9919523, 0.9976392, 0.9917915, 0.9978572, -0.0035516, 0.0034628
9: -0.0137388, -0.0085598, -0.0139422, -0.0084362, -0.0031345, 0.0032443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020201, upper bound: 0.0021460
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020187, upper bound: 0.0021323
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0063221, 0.0088328, 0.0062812, 0.0089675, -0.0016038, 0.0015549
1: 0.0022292, 0.0025978, 0.0022298, 0.0026178, -0.0002374, 0.0002237
2: 0.0094758, 0.0108646, 0.0094020, 0.0108871, -0.0008607, 0.0008867
3: -0.0048772, -0.0034373, -0.0049565, -0.0034205, -0.0008854, 0.0009228
4: -0.0003095, 0.0012429, -0.0003341, 0.0013288, -0.0009934, 0.0009585
5: 0.0029354, 0.0044056, 0.0028559, 0.0044295, -0.0009099, 0.0009395
6: -0.0106487, -0.0048202, -0.0109690, -0.0047253, -0.0036022, 0.0037277
7: 0.0040005, 0.0119432, 0.0038788, 0.0123821, -0.0050834, 0.0049015
8: 0.9920372, 0.9976275, 0.9919462, 0.9979360, -0.0035762, 0.0034538
9: -0.0137331, -0.0086567, -0.0140138, -0.0085765, -0.0031342, 0.0032484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020201, upper bound: 0.0021363
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020189, upper bound: 0.0021169
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0063203, 0.0088428, 0.0062686, 0.0089898, -0.0016347, 0.0015847
1: 0.0022171, 0.0025982, 0.0022279, 0.0026211, -0.0002531, 0.0002262
2: 0.0094690, 0.0108656, 0.0093896, 0.0108941, -0.0008794, 0.0009038
3: -0.0048787, -0.0034244, -0.0049693, -0.0034133, -0.0008954, 0.0009512
4: -0.0003118, 0.0012445, -0.0003419, 0.0013426, -0.0010136, 0.0009693
5: 0.0029307, 0.0044067, 0.0028428, 0.0044369, -0.0009253, 0.0009576
6: -0.0106581, -0.0048160, -0.0110209, -0.0046960, -0.0036486, 0.0037995
7: 0.0039809, 0.0119511, 0.0038389, 0.0124527, -0.0051936, 0.0049567
8: 0.9920332, 0.9976344, 0.9919181, 0.9979858, -0.0036451, 0.0034946
9: -0.0137382, -0.0086486, -0.0140590, -0.0085510, -0.0031695, 0.0033150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020201, upper bound: 0.0021359
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020187, upper bound: 0.0021175
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0062378, 0.0091564, 0.0062271, 0.0089093, -0.0017110, 0.0021695
1: 0.0018814, 0.0026120, 0.0022219, 0.0026094, -0.0006118, 0.0002475
2: 0.0092428, 0.0109111, 0.0094342, 0.0109171, -0.0013074, 0.0009460
3: -0.0049335, -0.0030730, -0.0049232, -0.0033895, -0.0009797, 0.0013195
4: -0.0003925, 0.0013038, -0.0003676, 0.0012927, -0.0010900, 0.0010606
5: 0.0027887, 0.0044549, 0.0028900, 0.0044612, -0.0011789, 0.0010023
6: -0.0109672, -0.0046245, -0.0108336, -0.0045995, -0.0041319, 0.0039769
7: 0.0033795, 0.0122547, 0.0037074, 0.0121978, -0.0057796, 0.0054235
8: 0.9918494, 0.9978746, 0.9918255, 0.9978063, -0.0038153, 0.0038624
9: -0.0139323, -0.0083707, -0.0138959, -0.0084670, -0.0034680, 0.0035790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020818, upper bound: 0.0021374
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020685, upper bound: 0.0021285
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0062359, 0.0091740, 0.0062119, 0.0089321, -0.0017458, 0.0022159
1: 0.0018632, 0.0026124, 0.0022197, 0.0026127, -0.0006358, 0.0002498
2: 0.0092275, 0.0109122, 0.0094215, 0.0109255, -0.0013409, 0.0009652
3: -0.0049350, -0.0030551, -0.0049363, -0.0033808, -0.0009888, 0.0013564
4: -0.0003951, 0.0013055, -0.0003770, 0.0013069, -0.0011131, 0.0010705
5: 0.0027814, 0.0044561, 0.0028766, 0.0044701, -0.0011997, 0.0010227
6: -0.0109785, -0.0046200, -0.0108867, -0.0045641, -0.0041776, 0.0040577
7: 0.0033562, 0.0122632, 0.0036592, 0.0122701, -0.0059096, 0.0054741
8: 0.9918451, 0.9978821, 0.9917915, 0.9978572, -0.0038928, 0.0039002
9: -0.0139378, -0.0083614, -0.0139422, -0.0084362, -0.0035003, 0.0036552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020818, upper bound: 0.0021380
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020685, upper bound: 0.0021309
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0062739, 0.0091151, 0.0062812, 0.0089675, -0.0017647, 0.0020921
1: 0.0019373, 0.0026120, 0.0022298, 0.0026178, -0.0005628, 0.0002496
2: 0.0092791, 0.0108912, 0.0094020, 0.0108871, -0.0012430, 0.0009757
3: -0.0049333, -0.0031408, -0.0049565, -0.0034205, -0.0009878, 0.0012950
4: -0.0003658, 0.0013036, -0.0003341, 0.0013288, -0.0011185, 0.0010693
5: 0.0028043, 0.0044338, 0.0028559, 0.0044295, -0.0011537, 0.0010338
6: -0.0109522, -0.0047082, -0.0109690, -0.0047253, -0.0041375, 0.0041016
7: 0.0035431, 0.0122536, 0.0038788, 0.0123821, -0.0058936, 0.0054681
8: 0.9919297, 0.9978698, 0.9919462, 0.9979360, -0.0039349, 0.0038871
9: -0.0139316, -0.0084597, -0.0140138, -0.0085765, -0.0034965, 0.0036701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020749, upper bound: 0.0021361
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020665, upper bound: 0.0021169
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0062708, 0.0091306, 0.0062686, 0.0089898, -0.0017908, 0.0021360
1: 0.0019208, 0.0026124, 0.0022279, 0.0026211, -0.0005842, 0.0002522
2: 0.0092668, 0.0108929, 0.0093896, 0.0108941, -0.0012740, 0.0009901
3: -0.0049348, -0.0031248, -0.0049693, -0.0034133, -0.0009981, 0.0013271
4: -0.0003687, 0.0013053, -0.0003419, 0.0013426, -0.0011365, 0.0010805
5: 0.0027979, 0.0044356, 0.0028428, 0.0044369, -0.0011735, 0.0010491
6: -0.0109629, -0.0047011, -0.0110209, -0.0046960, -0.0041870, 0.0041624
7: 0.0035227, 0.0122620, 0.0038389, 0.0124527, -0.0059929, 0.0055251
8: 0.9919230, 0.9978770, 0.9919181, 0.9979858, -0.0039932, 0.0039292
9: -0.0139370, -0.0084501, -0.0140590, -0.0085510, -0.0035329, 0.0037298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020749, upper bound: 0.0021356
time: 0.83 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020665, upper bound: 0.0021158
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0062157, 0.0090716, 0.0060880, 0.0089095, -0.0016746, 0.0021892
1: 0.0018943, 0.0025979, 0.0022018, 0.0026095, -0.0005940, 0.0002265
2: 0.0092870, 0.0109233, 0.0094341, 0.0109940, -0.0013378, 0.0009258
3: -0.0048775, -0.0030724, -0.0049233, -0.0033100, -0.0008964, 0.0012699
4: -0.0004051, 0.0012432, -0.0004537, 0.0012928, -0.0010655, 0.0009704
5: 0.0028407, 0.0044679, 0.0028899, 0.0045427, -0.0011664, 0.0009810
6: -0.0107408, -0.0045731, -0.0108341, -0.0042762, -0.0038562, 0.0038921
7: 0.0033224, 0.0119444, 0.0032671, 0.0121983, -0.0056386, 0.0049625
8: 0.9918001, 0.9976574, 0.9915153, 0.9978066, -0.0037340, 0.0035559
9: -0.0137339, -0.0083299, -0.0138963, -0.0081854, -0.0031731, 0.0034981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021229, upper bound: 0.0022886
time: 0.72 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021008, upper bound: 0.0022778
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0062136, 0.0090913, 0.0060709, 0.0089409, -0.0017224, 0.0022515
1: 0.0018747, 0.0025982, 0.0021911, 0.0026131, -0.0006197, 0.0002373
2: 0.0092706, 0.0109245, 0.0094156, 0.0110034, -0.0013809, 0.0009540
3: -0.0048789, -0.0030540, -0.0049376, -0.0032919, -0.0009161, 0.0013121
4: -0.0004077, 0.0012447, -0.0004651, 0.0013083, -0.0010907, 0.0009839
5: 0.0028331, 0.0044691, 0.0028725, 0.0045527, -0.0011948, 0.0010074
6: -0.0107522, -0.0045681, -0.0108951, -0.0042365, -0.0039164, 0.0039848
7: 0.0032970, 0.0119525, 0.0032033, 0.0122772, -0.0057752, 0.0050366
8: 0.9917952, 0.9976647, 0.9914771, 0.9978632, -0.0038197, 0.0036050
9: -0.0137391, -0.0083206, -0.0139468, -0.0081476, -0.0032176, 0.0035812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021229, upper bound: 0.0022893
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021007, upper bound: 0.0022811
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0062542, 0.0090278, 0.0061317, 0.0089728, -0.0017130, 0.0020998
1: 0.0019477, 0.0025978, 0.0022081, 0.0026186, -0.0005456, 0.0002294
2: 0.0093237, 0.0109021, 0.0093990, 0.0109698, -0.0012674, 0.0009471
3: -0.0048773, -0.0031407, -0.0049595, -0.0033350, -0.0009081, 0.0012416
4: -0.0003769, 0.0012430, -0.0004267, 0.0013320, -0.0010846, 0.0009831
5: 0.0028566, 0.0044453, 0.0028528, 0.0045171, -0.0011347, 0.0010035
6: -0.0107257, -0.0046626, -0.0109812, -0.0043777, -0.0038699, 0.0039816
7: 0.0034945, 0.0119435, 0.0034053, 0.0123987, -0.0057073, 0.0050273
8: 0.9918859, 0.9976525, 0.9916127, 0.9979478, -0.0038198, 0.0035926
9: -0.0137333, -0.0084243, -0.0140244, -0.0082738, -0.0032146, 0.0035581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021012, upper bound: 0.0022831
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021008, upper bound: 0.0022641
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0090463, 0.0061183, 0.0089947, -0.0017438, 0.0021602
1: 0.0019292, 0.0025982, 0.0022062, 0.0026218, -0.0005706, 0.0002325
2: 0.0093093, 0.0109035, 0.0093869, 0.0109772, -0.0013089, 0.0009641
3: -0.0048787, -0.0031231, -0.0049720, -0.0033273, -0.0009201, 0.0012801
4: -0.0003799, 0.0012446, -0.0004350, 0.0013456, -0.0011058, 0.0009961
5: 0.0028491, 0.0044469, 0.0028400, 0.0045250, -0.0011626, 0.0010215
6: -0.0107369, -0.0046564, -0.0110321, -0.0043466, -0.0039308, 0.0040531
7: 0.0034672, 0.0119515, 0.0033629, 0.0124681, -0.0058246, 0.0050937
8: 0.9918801, 0.9976598, 0.9915828, 0.9979966, -0.0038884, 0.0036425
9: -0.0137384, -0.0084135, -0.0140688, -0.0082467, -0.0032570, 0.0036284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021012, upper bound: 0.0022860
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021007, upper bound: 0.0022680
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061681, 0.0094527, 0.0060880, 0.0089095, -0.0018264, 0.0028364
1: 0.0015885, 0.0026121, 0.0022018, 0.0026095, -0.0009372, 0.0002548
2: 0.0089859, 0.0109497, 0.0094341, 0.0109940, -0.0018099, 0.0010098
3: -0.0049336, -0.0027566, -0.0049233, -0.0033100, -0.0010083, 0.0016851
4: -0.0004614, 0.0013039, -0.0004537, 0.0012928, -0.0011885, 0.0010916
5: 0.0026883, 0.0044958, 0.0028899, 0.0045427, -0.0014483, 0.0010699
6: -0.0110443, -0.0044623, -0.0108341, -0.0042762, -0.0044071, 0.0042450
7: 0.0028614, 0.0122552, 0.0032671, 0.0121983, -0.0064703, 0.0055819
8: 0.9916939, 0.9978982, 0.9915153, 0.9978066, -0.0040725, 0.0040190
9: -0.0139326, -0.0081329, -0.0138963, -0.0081854, -0.0035692, 0.0039152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021748, upper bound: 0.0022822
time: 0.82 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021617, upper bound: 0.0022762
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0061632, 0.0094771, 0.0060709, 0.0089409, -0.0018694, 0.0028963
1: 0.0015699, 0.0026124, 0.0021911, 0.0026131, -0.0009602, 0.0002657
2: 0.0089673, 0.0109524, 0.0094156, 0.0110034, -0.0018506, 0.0010352
3: -0.0049351, -0.0027369, -0.0049376, -0.0032919, -0.0010283, 0.0017205
4: -0.0004650, 0.0013056, -0.0004651, 0.0013083, -0.0012096, 0.0011054
5: 0.0026791, 0.0044986, 0.0028725, 0.0045527, -0.0014757, 0.0010935
6: -0.0110556, -0.0044510, -0.0108951, -0.0042365, -0.0044656, 0.0043262
7: 0.0028337, 0.0122638, 0.0032033, 0.0122772, -0.0065820, 0.0056578
8: 0.9916830, 0.9979059, 0.9914771, 0.9978632, -0.0041473, 0.0040683
9: -0.0139381, -0.0081209, -0.0139468, -0.0081476, -0.0036149, 0.0039838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021748, upper bound: 0.0022822
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021617, upper bound: 0.0022752
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0062023, 0.0093841, 0.0061317, 0.0089728, -0.0018707, 0.0027289
1: 0.0016514, 0.0026120, 0.0022081, 0.0026186, -0.0008799, 0.0002577
2: 0.0090434, 0.0109308, 0.0093990, 0.0109698, -0.0017249, 0.0010343
3: -0.0049334, -0.0028304, -0.0049595, -0.0033350, -0.0010198, 0.0016479
4: -0.0004353, 0.0013037, -0.0004267, 0.0013320, -0.0012105, 0.0011040
5: 0.0027117, 0.0044757, 0.0028528, 0.0045171, -0.0014086, 0.0010959
6: -0.0110280, -0.0045419, -0.0109812, -0.0043777, -0.0044192, 0.0043481
7: 0.0030279, 0.0122540, 0.0034053, 0.0123987, -0.0065398, 0.0056454
8: 0.9917702, 0.9978925, 0.9916127, 0.9979478, -0.0041714, 0.0040541
9: -0.0139319, -0.0082207, -0.0140244, -0.0082738, -0.0036098, 0.0039840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021723, upper bound: 0.0022670
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021594, upper bound: 0.0022591
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0061983, 0.0094064, 0.0061183, 0.0089947, -0.0018944, 0.0027888
1: 0.0016337, 0.0026124, 0.0022062, 0.0026218, -0.0009024, 0.0002608
2: 0.0090248, 0.0109330, 0.0093869, 0.0109772, -0.0017658, 0.0010474
3: -0.0049349, -0.0028121, -0.0049720, -0.0033273, -0.0010320, 0.0016813
4: -0.0004389, 0.0013054, -0.0004350, 0.0013456, -0.0012271, 0.0011173
5: 0.0027032, 0.0044781, 0.0028400, 0.0045250, -0.0014358, 0.0011097
6: -0.0110391, -0.0045326, -0.0110321, -0.0043466, -0.0044784, 0.0044031
7: 0.0029994, 0.0122625, 0.0033629, 0.0124681, -0.0066360, 0.0057133
8: 0.9917613, 0.9979000, 0.9915828, 0.9979966, -0.0042241, 0.0041044
9: -0.0139373, -0.0082082, -0.0140688, -0.0082467, -0.0036532, 0.0040392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021723, upper bound: 0.0022671
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021594, upper bound: 0.0022593
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062157, 0.0090716, 0.0063052, 0.0090972, -0.0019816, 0.0020099
1: 0.0018943, 0.0025979, 0.0019502, 0.0026103, -0.0005924, 0.0005028
2: 0.0092870, 0.0109233, 0.0092905, 0.0108739, -0.0012387, 0.0011757
3: -0.0048775, -0.0030724, -0.0049267, -0.0031659, -0.0010768, 0.0012636
4: -0.0004051, 0.0012432, -0.0003455, 0.0012964, -0.0010586, 0.0008860
5: 0.0028407, 0.0044679, 0.0028135, 0.0044155, -0.0010614, 0.0010950
6: -0.0107408, -0.0045731, -0.0109227, -0.0047810, -0.0034394, 0.0039771
7: 0.0033224, 0.0119444, 0.0036511, 0.0122169, -0.0056036, 0.0047003
8: 0.9918001, 0.9976574, 0.9919996, 0.9978430, -0.0037432, 0.0031561
9: -0.0137339, -0.0083299, -0.0139081, -0.0085268, -0.0029098, 0.0034757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021063, upper bound: 0.0021734
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021046, upper bound: 0.0021606
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061681, 0.0094527, 0.0063052, 0.0090972, -0.0019626, 0.0025820
1: 0.0015885, 0.0026121, 0.0019502, 0.0026103, -0.0008963, 0.0004891
2: 0.0089859, 0.0109497, 0.0092905, 0.0108739, -0.0016757, 0.0011813
3: -0.0049336, -0.0027566, -0.0049267, -0.0031659, -0.0010083, 0.0015095
4: -0.0004614, 0.0013039, -0.0003455, 0.0012964, -0.0010055, 0.0008240
5: 0.0026883, 0.0044958, 0.0028135, 0.0044155, -0.0012799, 0.0010653
6: -0.0110443, -0.0044623, -0.0109227, -0.0047810, -0.0034361, 0.0037119
7: 0.0028614, 0.0122552, 0.0036511, 0.0122169, -0.0055072, 0.0043758
8: 0.9916939, 0.9978982, 0.9919996, 0.9978430, -0.0034687, 0.0029958
9: -0.0139326, -0.0081329, -0.0139081, -0.0085268, -0.0027064, 0.0033145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021063, upper bound: 0.0021624
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021046, upper bound: 0.0021527
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062136, 0.0090913, 0.0062834, 0.0091638, -0.0021114, 0.0020718
1: 0.0018747, 0.0025982, 0.0018852, 0.0026138, -0.0006192, 0.0005735
2: 0.0092706, 0.0109245, 0.0092392, 0.0108860, -0.0012815, 0.0012729
3: -0.0048789, -0.0030540, -0.0049405, -0.0030935, -0.0011543, 0.0013101
4: -0.0004077, 0.0012447, -0.0003640, 0.0013114, -0.0010886, 0.0009036
5: 0.0028331, 0.0044691, 0.0027831, 0.0044283, -0.0010896, 0.0011512
6: -0.0107522, -0.0045681, -0.0109965, -0.0047302, -0.0034989, 0.0041089
7: 0.0032970, 0.0119525, 0.0035260, 0.0122932, -0.0057642, 0.0048298
8: 0.9917952, 0.9976647, 0.9919509, 0.9979016, -0.0038523, 0.0032044
9: -0.0137391, -0.0083206, -0.0139569, -0.0084639, -0.0029696, 0.0035742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021063, upper bound: 0.0021843
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021046, upper bound: 0.0021750
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061632, 0.0094771, 0.0062834, 0.0091638, -0.0021117, 0.0026464
1: 0.0015699, 0.0026124, 0.0018852, 0.0026138, -0.0009214, 0.0005609
2: 0.0089673, 0.0109524, 0.0092392, 0.0108860, -0.0017187, 0.0012892
3: -0.0049351, -0.0027369, -0.0049405, -0.0030935, -0.0010880, 0.0015552
4: -0.0004650, 0.0013056, -0.0003640, 0.0013114, -0.0010349, 0.0008432
5: 0.0026791, 0.0044986, 0.0027831, 0.0044283, -0.0013107, 0.0011340
6: -0.0110556, -0.0044510, -0.0109965, -0.0047302, -0.0035023, 0.0038513
7: 0.0028337, 0.0122638, 0.0035260, 0.0122932, -0.0056620, 0.0045112
8: 0.9916830, 0.9979059, 0.9919509, 0.9979016, -0.0035763, 0.0030491
9: -0.0139381, -0.0081209, -0.0139569, -0.0084639, -0.0027717, 0.0034111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021268, upper bound: 0.0021498
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021046, upper bound: 0.0021592
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062542, 0.0090278, 0.0063490, 0.0090867, -0.0019145, 0.0019162
1: 0.0019477, 0.0025978, 0.0020497, 0.0026195, -0.0005484, 0.0004043
2: 0.0093237, 0.0109021, 0.0093137, 0.0108497, -0.0011659, 0.0010997
3: -0.0048773, -0.0031407, -0.0049631, -0.0032785, -0.0009923, 0.0012524
4: -0.0003769, 0.0012430, -0.0003102, 0.0013359, -0.0010963, 0.0008875
5: 0.0028566, 0.0044453, 0.0028057, 0.0043898, -0.0010271, 0.0010891
6: -0.0107257, -0.0046626, -0.0110397, -0.0048828, -0.0034431, 0.0040998
7: 0.0034945, 0.0119435, 0.0038859, 0.0124185, -0.0057672, 0.0046537
8: 0.9918859, 0.9976525, 0.9920972, 0.9979753, -0.0038847, 0.0031832
9: -0.0137333, -0.0084243, -0.0140371, -0.0086461, -0.0029106, 0.0035964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021063, upper bound: 0.0021757
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021046, upper bound: 0.0021583
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062023, 0.0093841, 0.0063490, 0.0090867, -0.0018479, 0.0024698
1: 0.0016514, 0.0026120, 0.0020497, 0.0026195, -0.0008404, 0.0003892
2: 0.0090434, 0.0109308, 0.0093137, 0.0108497, -0.0015901, 0.0010800
3: -0.0049334, -0.0028304, -0.0049631, -0.0032785, -0.0009266, 0.0014759
4: -0.0004353, 0.0013037, -0.0003102, 0.0013359, -0.0010340, 0.0008307
5: 0.0027117, 0.0044757, 0.0028057, 0.0043898, -0.0012367, 0.0010323
6: -0.0110280, -0.0045419, -0.0110397, -0.0048828, -0.0034425, 0.0037898
7: 0.0030279, 0.0122540, 0.0038859, 0.0124185, -0.0056143, 0.0043547
8: 0.9917702, 0.9978925, 0.9920972, 0.9979753, -0.0035724, 0.0030385
9: -0.0139319, -0.0082207, -0.0140371, -0.0086461, -0.0027243, 0.0034051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021063, upper bound: 0.0021492
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021046, upper bound: 0.0021308
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0090463, 0.0063314, 0.0091423, -0.0020145, 0.0019749
1: 0.0019292, 0.0025982, 0.0019931, 0.0026229, -0.0005746, 0.0004664
2: 0.0093093, 0.0109035, 0.0092747, 0.0108594, -0.0012064, 0.0011731
3: -0.0048787, -0.0031231, -0.0049766, -0.0032177, -0.0010599, 0.0012960
4: -0.0003799, 0.0012446, -0.0003260, 0.0013505, -0.0011231, 0.0009041
5: 0.0028491, 0.0044469, 0.0027773, 0.0044001, -0.0010540, 0.0011313
6: -0.0107369, -0.0046564, -0.0111076, -0.0048419, -0.0035000, 0.0042076
7: 0.0034672, 0.0119515, 0.0037741, 0.0124933, -0.0059129, 0.0047697
8: 0.9918801, 0.9976598, 0.9920580, 0.9980317, -0.0039773, 0.0032291
9: -0.0137384, -0.0084135, -0.0140849, -0.0085922, -0.0029671, 0.0036848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021063, upper bound: 0.0021842
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021046, upper bound: 0.0021694
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061983, 0.0094064, 0.0063314, 0.0091423, -0.0019748, 0.0025348
1: 0.0016337, 0.0026124, 0.0019931, 0.0026229, -0.0008645, 0.0004534
2: 0.0090248, 0.0109330, 0.0092747, 0.0108594, -0.0016333, 0.0011674
3: -0.0049349, -0.0028121, -0.0049766, -0.0032177, -0.0009954, 0.0015172
4: -0.0004389, 0.0013054, -0.0003260, 0.0013505, -0.0010575, 0.0008495
5: 0.0027032, 0.0044781, 0.0027773, 0.0044001, -0.0012675, 0.0010882
6: -0.0110391, -0.0045326, -0.0111076, -0.0048419, -0.0035095, 0.0038974
7: 0.0029994, 0.0122625, 0.0037741, 0.0124933, -0.0057440, 0.0044816
8: 0.9917613, 0.9979000, 0.9920580, 0.9980317, -0.0036615, 0.0030931
9: -0.0139373, -0.0082082, -0.0140849, -0.0085922, -0.0027879, 0.0034824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021063, upper bound: 0.0021545
time: 0.90 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0021046, upper bound: 0.0021383
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062858, 0.0088599, 0.0061844, 0.0090082, -0.0017015, 0.0017035
1: 0.0021810, 0.0025979, 0.0022158, 0.0026237, -0.0002935, 0.0002378
2: 0.0094559, 0.0108846, 0.0093795, 0.0109407, -0.0009535, 0.0009407
3: -0.0048774, -0.0033739, -0.0049798, -0.0033651, -0.0009413, 0.0010183
4: -0.0003362, 0.0012431, -0.0003940, 0.0013539, -0.0010577, 0.0010190
5: 0.0029238, 0.0044268, 0.0028321, 0.0044862, -0.0009872, 0.0009967
6: -0.0106622, -0.0047359, -0.0110635, -0.0045003, -0.0038503, 0.0039546
7: 0.0038362, 0.0119441, 0.0035723, 0.0125108, -0.0054376, 0.0052107
8: 0.9919564, 0.9976324, 0.9917302, 0.9980267, -0.0037939, 0.0036786
9: -0.0137337, -0.0085672, -0.0140961, -0.0083806, -0.0033319, 0.0034608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020175, upper bound: 0.0021725
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020159, upper bound: 0.0021585
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062378, 0.0091564, 0.0061844, 0.0090082, -0.0016045, 0.0021655
1: 0.0018814, 0.0026120, 0.0022158, 0.0026237, -0.0005977, 0.0002287
2: 0.0092428, 0.0109111, 0.0093795, 0.0109407, -0.0013218, 0.0008871
3: -0.0049335, -0.0030730, -0.0049798, -0.0033651, -0.0009052, 0.0012484
4: -0.0003925, 0.0013038, -0.0003940, 0.0013539, -0.0010227, 0.0009799
5: 0.0027887, 0.0044549, 0.0028321, 0.0044862, -0.0011563, 0.0009399
6: -0.0109672, -0.0046245, -0.0110635, -0.0045003, -0.0038684, 0.0037293
7: 0.0033795, 0.0122547, 0.0035723, 0.0125108, -0.0054229, 0.0050108
8: 0.9918494, 0.9978746, 0.9917302, 0.9980267, -0.0035777, 0.0035839
9: -0.0139323, -0.0083707, -0.0140961, -0.0083806, -0.0032041, 0.0033583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020175, upper bound: 0.0021640
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020159, upper bound: 0.0021503
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062840, 0.0088696, 0.0061699, 0.0090293, -0.0017448, 0.0017337
1: 0.0021703, 0.0025982, 0.0022137, 0.0026268, -0.0003115, 0.0002401
2: 0.0094493, 0.0108856, 0.0093678, 0.0109487, -0.0009736, 0.0009646
3: -0.0048788, -0.0033633, -0.0049918, -0.0033568, -0.0009501, 0.0010529
4: -0.0003384, 0.0012447, -0.0004030, 0.0013670, -0.0010855, 0.0010286
5: 0.0029194, 0.0044279, 0.0028197, 0.0044947, -0.0010024, 0.0010221
6: -0.0106712, -0.0047317, -0.0111125, -0.0044666, -0.0038916, 0.0040553
7: 0.0038191, 0.0119520, 0.0035264, 0.0125776, -0.0055855, 0.0052598
8: 0.9919523, 0.9976392, 0.9916980, 0.9980738, -0.0038905, 0.0037149
9: -0.0137388, -0.0085598, -0.0141388, -0.0083512, -0.0033633, 0.0035519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020175, upper bound: 0.0021778
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020157, upper bound: 0.0021691
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062359, 0.0091740, 0.0061699, 0.0090293, -0.0016425, 0.0022195
1: 0.0018632, 0.0026124, 0.0022137, 0.0026268, -0.0006222, 0.0002311
2: 0.0092275, 0.0109122, 0.0093678, 0.0109487, -0.0013587, 0.0009081
3: -0.0049350, -0.0030551, -0.0049918, -0.0033568, -0.0009147, 0.0012874
4: -0.0003951, 0.0013055, -0.0004030, 0.0013670, -0.0010472, 0.0009902
5: 0.0027814, 0.0044561, 0.0028197, 0.0044947, -0.0011803, 0.0009622
6: -0.0109785, -0.0046200, -0.0111125, -0.0044666, -0.0039182, 0.0038176
7: 0.0033562, 0.0122632, 0.0035264, 0.0125776, -0.0055571, 0.0050637
8: 0.9918451, 0.9978821, 0.9916980, 0.9980738, -0.0036625, 0.0036240
9: -0.0139378, -0.0083614, -0.0141388, -0.0083512, -0.0032379, 0.0034391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020175, upper bound: 0.0021633
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020157, upper bound: 0.0021516
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063221, 0.0088328, 0.0062431, 0.0090668, -0.0017651, 0.0016654
1: 0.0022292, 0.0025978, 0.0022242, 0.0026322, -0.0002607, 0.0002397
2: 0.0094758, 0.0108646, 0.0093470, 0.0109082, -0.0009218, 0.0009759
3: -0.0048772, -0.0034373, -0.0050133, -0.0033987, -0.0009486, 0.0010150
4: -0.0003095, 0.0012429, -0.0003577, 0.0013902, -0.0010932, 0.0010269
5: 0.0029354, 0.0044056, 0.0027977, 0.0044518, -0.0009746, 0.0010340
6: -0.0106487, -0.0048202, -0.0111998, -0.0046367, -0.0038591, 0.0041026
7: 0.0040005, 0.0119432, 0.0037581, 0.0126965, -0.0055939, 0.0052515
8: 0.9920372, 0.9976275, 0.9918611, 0.9981575, -0.0039359, 0.0037003
9: -0.0137331, -0.0086567, -0.0142148, -0.0084994, -0.0033579, 0.0035749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020175, upper bound: 0.0021702
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020159, upper bound: 0.0021512
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062739, 0.0091151, 0.0062431, 0.0090668, -0.0016577, 0.0020731
1: 0.0019373, 0.0026120, 0.0022242, 0.0026322, -0.0005478, 0.0002310
2: 0.0092791, 0.0108912, 0.0093470, 0.0109082, -0.0012500, 0.0009165
3: -0.0049333, -0.0031408, -0.0050133, -0.0033987, -0.0009143, 0.0012217
4: -0.0003658, 0.0013036, -0.0003577, 0.0013902, -0.0010508, 0.0009898
5: 0.0028043, 0.0044338, 0.0027977, 0.0044518, -0.0011233, 0.0009711
6: -0.0109522, -0.0047082, -0.0111998, -0.0046367, -0.0038721, 0.0038529
7: 0.0035431, 0.0122536, 0.0037581, 0.0126965, -0.0055339, 0.0050613
8: 0.9919297, 0.9978698, 0.9918611, 0.9981575, -0.0036963, 0.0036110
9: -0.0139316, -0.0084597, -0.0142148, -0.0084994, -0.0032363, 0.0034476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020175, upper bound: 0.0021519
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020159, upper bound: 0.0021354
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063203, 0.0088428, 0.0062316, 0.0090880, -0.0017915, 0.0016952
1: 0.0022171, 0.0025982, 0.0022226, 0.0026353, -0.0002757, 0.0002422
2: 0.0094690, 0.0108656, 0.0093353, 0.0109146, -0.0009405, 0.0009905
3: -0.0048787, -0.0034244, -0.0050254, -0.0033921, -0.0009586, 0.0010408
4: -0.0003118, 0.0012445, -0.0003648, 0.0014033, -0.0011106, 0.0010377
5: 0.0029307, 0.0044067, 0.0027853, 0.0044586, -0.0009900, 0.0010495
6: -0.0106581, -0.0048160, -0.0112490, -0.0046099, -0.0039054, 0.0041639
7: 0.0039809, 0.0119511, 0.0037215, 0.0127635, -0.0056900, 0.0053064
8: 0.9920332, 0.9976344, 0.9918354, 0.9982048, -0.0039947, 0.0037410
9: -0.0137382, -0.0086486, -0.0142577, -0.0084760, -0.0033931, 0.0036324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020175, upper bound: 0.0021721
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020157, upper bound: 0.0021586
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062708, 0.0091306, 0.0062316, 0.0090880, -0.0016873, 0.0021230
1: 0.0019208, 0.0026124, 0.0022226, 0.0026353, -0.0005696, 0.0002335
2: 0.0092668, 0.0108929, 0.0093353, 0.0109146, -0.0012837, 0.0009329
3: -0.0049348, -0.0031248, -0.0050254, -0.0033921, -0.0009242, 0.0012561
4: -0.0003687, 0.0013053, -0.0003648, 0.0014033, -0.0010704, 0.0010005
5: 0.0027979, 0.0044356, 0.0027853, 0.0044586, -0.0011456, 0.0009884
6: -0.0109629, -0.0047011, -0.0112490, -0.0046099, -0.0039223, 0.0039218
7: 0.0035227, 0.0122620, 0.0037215, 0.0127635, -0.0056422, 0.0051165
8: 0.9919230, 0.9978770, 0.9918354, 0.9982048, -0.0037625, 0.0036523
9: -0.0139370, -0.0084501, -0.0142577, -0.0084760, -0.0032716, 0.0035124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020175, upper bound: 0.0021506
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020157, upper bound: 0.0021330
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062157, 0.0090716, 0.0060425, 0.0091640, -0.0020288, 0.0023029
1: 0.0018943, 0.0025979, 0.0020043, 0.0026243, -0.0006140, 0.0004461
2: 0.0092870, 0.0109233, 0.0092645, 0.0110191, -0.0014007, 0.0011669
3: -0.0048775, -0.0030724, -0.0049820, -0.0031107, -0.0011485, 0.0013492
4: -0.0004051, 0.0012432, -0.0004990, 0.0013563, -0.0011514, 0.0010591
5: 0.0028407, 0.0044679, 0.0027669, 0.0045694, -0.0012330, 0.0011510
6: -0.0107408, -0.0045731, -0.0111391, -0.0041705, -0.0041203, 0.0043102
7: 0.0033224, 0.0119444, 0.0029267, 0.0125231, -0.0060777, 0.0055332
8: 0.9918001, 0.9976574, 0.9914139, 0.9980567, -0.0040738, 0.0038094
9: -0.0137339, -0.0083299, -0.0141039, -0.0080291, -0.0034719, 0.0037788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020986, upper bound: 0.0023342
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020987, upper bound: 0.0023244
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061681, 0.0094527, 0.0060425, 0.0091640, -0.0019992, 0.0029164
1: 0.0015885, 0.0026121, 0.0020043, 0.0026243, -0.0009237, 0.0004338
2: 0.0089859, 0.0109497, 0.0092645, 0.0110191, -0.0018606, 0.0011656
3: -0.0049336, -0.0027566, -0.0049820, -0.0031107, -0.0011050, 0.0016183
4: -0.0004614, 0.0013039, -0.0004990, 0.0013563, -0.0011232, 0.0010233
5: 0.0026883, 0.0044958, 0.0027669, 0.0045694, -0.0014758, 0.0011188
6: -0.0110443, -0.0044623, -0.0111391, -0.0041705, -0.0042133, 0.0041157
7: 0.0028614, 0.0122552, 0.0029267, 0.0125231, -0.0061095, 0.0053408
8: 0.9916939, 0.9978982, 0.9914139, 0.9980567, -0.0038824, 0.0037415
9: -0.0139326, -0.0081329, -0.0141039, -0.0080291, -0.0033542, 0.0036996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021219, upper bound: 0.0023048
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020987, upper bound: 0.0023089
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062136, 0.0090913, 0.0060258, 0.0092220, -0.0021286, 0.0023629
1: 0.0018747, 0.0025982, 0.0019526, 0.0026279, -0.0006408, 0.0005026
2: 0.0092706, 0.0109245, 0.0092252, 0.0110284, -0.0014425, 0.0012383
3: -0.0048789, -0.0030540, -0.0049965, -0.0030558, -0.0012081, 0.0013956
4: -0.0004077, 0.0012447, -0.0005138, 0.0013720, -0.0011811, 0.0010749
5: 0.0028331, 0.0044691, 0.0027377, 0.0045792, -0.0012601, 0.0011968
6: -0.0107522, -0.0045681, -0.0112100, -0.0041315, -0.0041754, 0.0044277
7: 0.0032970, 0.0119525, 0.0028232, 0.0126033, -0.0062374, 0.0056454
8: 0.9917952, 0.9976647, 0.9913765, 0.9981170, -0.0041788, 0.0038535
9: -0.0137391, -0.0083206, -0.0141553, -0.0079787, -0.0035259, 0.0038768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020986, upper bound: 0.0023467
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020987, upper bound: 0.0023407
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061632, 0.0094771, 0.0060258, 0.0092220, -0.0021131, 0.0029783
1: 0.0015699, 0.0026124, 0.0019526, 0.0026279, -0.0009474, 0.0004911
2: 0.0089673, 0.0109524, 0.0092252, 0.0110284, -0.0019022, 0.0012440
3: -0.0049351, -0.0027369, -0.0049965, -0.0030558, -0.0011627, 0.0016583
4: -0.0004650, 0.0013056, -0.0005138, 0.0013720, -0.0011464, 0.0010399
5: 0.0026791, 0.0044986, 0.0027377, 0.0045792, -0.0015051, 0.0011709
6: -0.0110556, -0.0044510, -0.0112100, -0.0041315, -0.0042737, 0.0042275
7: 0.0028337, 0.0122638, 0.0028232, 0.0126033, -0.0062323, 0.0054533
8: 0.9916830, 0.9979059, 0.9913765, 0.9981170, -0.0039697, 0.0037891
9: -0.0139381, -0.0081209, -0.0141553, -0.0079787, -0.0034102, 0.0037757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.45 + 597.37 = 600.82 seconds
